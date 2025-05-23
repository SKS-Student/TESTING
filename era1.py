import asyncio  # Import asyncio for asynchronous programming
import cv2  # OpenCV for computer vision tasks (image and video handling)
import depthai as dai  # DepthAI library for interfacing with stereo cameras
import numpy as np  # NumPy for numerical operations
import pygame  # Pygame for handling audio playback (text-to-speech)
import os  # OS module for interacting with the operating system (e.g., file operations)
import tempfile  # For creating temporary files
import edge_tts  # For text-to-speech functionality (Edge TTS)
import speech_recognition as sr  # Speech recognition library
import random  # For generating random numbers (used for jokes)
import pathlib  # Path handling library
import torch  # PyTorch library (though not used directly here)
import time  # For time-related functions
from threading import Thread  # For creating threads to run concurrent tasks
from transformers import AutoModelForCausalLM, AutoTokenizer  # Transformers library for using pre-trained language models

# Load pre-trained DialoGPT model and tokenizer for chatbot functionality
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialization of various variables and devices
pygame.mixer.init()  # Initialize pygame mixer for audio playback
recognizer = sr.Recognizer()  # Initialize the speech recognizer
audio_available = True  # Flag to indicate if audio is available for TTS
name_of_user = "User"  # Placeholder for the user's name
interrupt_event = False  # Flag for interrupting audio playback
detecting = None  # Placeholder for the detected object info
label_map = [  # List of object labels for object detection
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

# ────────────────────────────────────────────────
# Text-to-Speech using Edge TTS
# ────────────────────────────────────────────────
# Converts the input text to speech and saves it as an MP3 file
async def get_speech(text):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")  # Initialize Edge TTS with the given text and voice model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:  # Create a temporary MP3 file
        filename = tmp.name  # Get the name of the temporary file
    await communicate.save(filename)  # Save the speech to the temporary file
    return filename  # Return the file path of the MP3

# Function to play the generated speech from the bot's response
async def speak_text(bot_response):
    global interrupt_event  # Access the global interrupt_event variable
    print(f"[DEBUG] TTS input: {repr(bot_response)}")  # Print the bot's response for debugging
    filename = await get_speech(bot_response)  # Convert the bot's response to speech and get the filename
    if audio_available:  # If audio is available
        pygame.mixer.music.load(filename)  # Load the MP3 file
        pygame.mixer.music.play()  # Play the MP3 file
        while pygame.mixer.music.get_busy():  # While the audio is still playing
            if interrupt_event:  # If the interrupt event is triggered
                pygame.mixer.music.stop()  # Stop the audio playback
                break  # Exit the loop
            pygame.time.Clock().tick(10)  # Wait for a short period before checking again
        os.remove(filename)  # Remove the temporary MP3 file after playback

# ────────────────────────────────────────────────
# Speech Recognition
# ────────────────────────────────────────────────
# Listens to the user's speech and converts it to text
def listen_to_user():
    with sr.Microphone() as source:  # Use the microphone as the audio source
        recognizer.adjust_for_ambient_noise(source, duration=0.8)  # Adjust for ambient noise
        print("Listening...")  # Print that the system is listening
        try:
            audio = recognizer.listen(source, timeout=7)  # Capture audio with a 7-second timeout
            text = recognizer.recognize_google(audio).lower()  # Convert audio to text using Google's API
            print(f"You said: {text}")  # Print the recognized text
            return text  # Return the recognized text
        except:
            return None  # If no speech was recognized, return None

# Listens for the user's name
def name_listener():
    with sr.Microphone() as source:  # Use the microphone to capture speech
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise
        print("Say your name...")  # Prompt the user to say their name
        try:
            audio = recognizer.listen(source, timeout=7)  # Capture audio with a 7-second timeout
            return recognizer.recognize_google(audio).lower(), True  # Return the name as text
        except:
            return None, False  # If no name is recognized, return None

# ────────────────────────────────────────────────
# Response Generation
# ────────────────────────────────────────────────
# Generates a response to the user's input, including jokes and predefined responses
def generate_response(user_input):
    user_input = user_input.lower()  # Convert the user's input to lowercase
    jokes = [  # List of predefined jokes
        "Why don't skeletons fight each other? They don't have the guts.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Why don’t oysters donate to charity? Because they are shellfish.",
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "What did the ocean say to the beach? Nothing, it just waved."
    ]
    if "who are you" in user_input:  # If the user asks who the assistant is
        return "I am Wonder, your personal AI Assistant"  # Return the assistant's introduction
    elif "a joke" in user_input:  # If the user asks for a joke
        return random.choice(jokes)  # Return a random joke from the list
    elif "hey wonder" in user_input:  # If the user says "Hey Wonder"
        return "Yes?"  # Respond with "Yes?"
    elif "change my name" in user_input:  # If the user asks to change their name
        return "name has been updated"  # Acknowledgment that the name has been changed
    elif "exit" in user_input:  # If the user wants to exit
        return f"Goodbye {name_of_user}! I will shut off now."  # Goodbye message
    elif "detect" in user_input:  # If the user asks to start object detection
        return("Detecting")  # Respond with a message saying detection has started
    elif "help" in user_input:  # If the user asks for help
        return("Say 'who are you' to get a description of this device. \n say 'a joke' to get a joke. \n Say 'change my name' to change the name we refer to you as.\n say 'detect' to see whats in front of you.\n Say 'exit' to stop running the device.")  # Provide help instructions
    else:
        # Use DialoGPT model to generate a response to the user's input
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')  # Tokenize the input
        bot_output = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, 
                                    no_repeat_ngram_size=3, top_p=0.92,  temperature=0.7)  # Generate a response using the model

        # Decode the response and print it
        bot_response = tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {bot_response}")  # Print the generated response
        return bot_response  # Return the bot's response

# ────────────────────────────────────────────────
# Disparity to Distance
# ────────────────────────────────────────────────
# Converts disparity values to actual distance measurements
def disparity_to_distance(disparities):
    valid_disparities = [d for d in disparities if d > 0]  # Filter out invalid disparities
    if not valid_disparities:  # If no valid disparities, return infinite distance
        return float('inf')
    avg_disparity = np.mean(valid_disparities)  # Calculate the average disparity
    distance = 140.99287 - 28.55858 * np.log(avg_disparity)  # Use the disparity to calculate distance
    return round(distance, 2)  # Return the calculated distance, rounded to 2 decimal places

# ────────────────────────────────────────────────
# Camera Thread
# ────────────────────────────────────────────────
# Thread for handling camera and object detection using DepthAI
def camera_thread():
    global interrupt_event, detecting  # Access the global interrupt_event and detecting variables
    pipeline = dai.Pipeline()  # Initialize a DepthAI pipeline
    # Set up the stereo camera and neural network (for object detection)
    ...
    with dai.Device(pipeline) as device:  # Start the DepthAI device with the pipeline
        qRight = device.getOutputQueue("rectifiedRight", 4, False)  # Get the queue for the right camera frame
        qDisparity = device.getOutputQueue("disparity", 4, False)  # Get the queue for disparity (depth)
        qDet = device.getOutputQueue("nn", 4, False)  # Get the queue for object detection

        while True:
            rightFrame = qRight.get().getCvFrame() if qRight.has() else None  # Capture the right camera frame
            disparityFrame = qDisparity.get().getFrame() if qDisparity.has() else None  # Capture the disparity frame
            detections = qDet.get().detections if qDet.has() else []  # Capture any detected objects
            ...
            if closest_label and closest_distance < 20:  # If an object is detected within 20 inches
                interrupt_event = True  # Trigger the interrupt event to stop speech
                detecting = (f"{closest_label} is {closest_distance} inches in front of you")  # Update the detecting info
                interrupt_event = False  # Reset the interrupt event

# ────────────────────────────────────────────────
# Chat Thread
# ────────────────────────────────────────────────
# Handles the user's name and chatbot conversation
async def chat_thread():
    global name_of_user  # Use the global variable `name_of_user` to store the user's name
    
    # Ask the user to say their name until it's recognized correctly
    while True:
        await speak_text("Say your name please")  # Prompt the user to say their name
        name_of_user, valid = name_listener()  # Use the `name_listener()` function to listen for the name
        
        if valid:  # If the name is recognized correctly
            await speak_text(f"Hey {name_of_user}")  # Greet the user with their name
            break  # Exit the loop once the name is successfully recognized
    
    # Main loop where the assistant waits for commands and interacts with the user
    while True:
        await speak_text("Ready for command")  # Inform the user that the system is ready for a command
        
        user_input = listen_to_user()  # Listen for the user's command using `listen_to_user()` function
        
        if not user_input:  # If no input is detected
            await speak_text("Say 'help' if you need the list of commands you can say")  # Prompt user for a command
            continue  # Continue to the next iteration of the loop, waiting for user input again
        
        print(f"{name_of_user} said: {user_input}")  # Print the recognized user input for debugging purposes
        
        response = generate_response(user_input)  # Generate the bot's response using the `generate_response()` function
        
        # If the user requests a name change
        if "name has been updated" in response:
            await speak_text("Say your new name")  # Ask the user to say their new name
            
            while True:
                name_of_user, valid = name_listener()  # Listen for the new name
                if valid:  # If a valid name is recognized
                    await speak_text(f"Hey {name_of_user}")  # Greet the user with their new name
                    break  # Exit the loop once the new name is recognized
        
        # If the user asks for object detection to be triggered
        elif "detect" in user_input:
            await speak_text(detecting if detecting else "Nothing close detected yet.")  # Speak out the detection result
            
        else:  # For other commands
            await speak_text(f"{response}")  # Respond with the bot's generated response

        # If the user wants to exit the program
        if "exit" in user_input:
            cv2.destroyAllWindows()  # Close any OpenCV windows that might be open
            os._exit(0)  # Terminate the program

# ────────────────────────────────────────────────
# Launch Threads
# ────────────────────────────────────────────────
if __name__ == "__main__":  # Ensures this block of code only runs if the script is executed directly (not imported as a module)
    # Start the assistant greeting and prompt
    asyncio.run(speak_text("Hello! I am Wonder, your personal AI assistant. I can help you with various tasks. Please wait 5 seconds for devices to pair"))
    
    # Create and start the camera thread (runs the `camera_thread()` function concurrently)
    cam_thread = Thread(target=camera_thread, daemon=True)  # Create a new thread for camera
    cam_thread.start()  # Start the camera thread

    # Inform the user to wait for devices to pair
    print("⏳ Waiting 5 seconds for devices to pair...")
    for i in range(5):  # Wait for 5 seconds
        time.sleep(1)  # Sleep for 1 second each iteration
        print(f"Startup: {i + 1}s")  # Print out the countdown each second

    # Run the chat thread which handles user interactions
    asyncio.run(chat_thread())  # Run the chatbot using asyncio for asynchronous tasks

