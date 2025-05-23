import asyncio  # For asynchronous programming
import cv2  # For handling computer vision tasks (camera, image processing)
import depthai as dai  # For working with DepthAI devices (stereo camera, object detection)
import numpy as np  # For numerical operations (e.g., handling image data)
import pygame  # For playing sounds (text-to-speech)
import os  # For interacting with the operating system (e.g., file management)
import tempfile  # For creating temporary files (used for TTS file saving)
import edge_tts  # For text-to-speech functionality
import speech_recognition as sr  # For speech-to-text functionality
import random  # For generating random responses (jokes)
import pathlib  # For managing file paths
import torch  # For using machine learning models (though not actively used in the code)

import time  # For time-related operations (e.g., delays, timers)
from threading import Thread  # For creating threads (to run multiple tasks concurrently)

from transformers import AutoModelForCausalLM, AutoTokenizer  # For loading the chatbot model from Hugging Face

# Load the DialoGPT model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# ────────────────────────────────────────────────
# Initialization
# ────────────────────────────────────────────────
pygame.mixer.init()  # Initialize the pygame mixer for audio playback
recognizer = sr.Recognizer()  # Initialize the speech recognition object
audio_available = True  # Flag to track if audio is available
name_of_user = "User"  # Default name of the user
interrupt_event = False  # Flag for interrupting music playback
detecting = None  # Stores the status of object detection (e.g., "Nothing detected yet")
label_map = [  # A list of labels for object detection (from the MobileNet model)
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

# ────────────────────────────────────────────────
# Text-to-Speech using Edge TTS
# ────────────────────────────────────────────────

# Function to get speech audio from text
async def get_speech(text):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")  # Initialize the TTS engine
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:  # Create a temporary file to save the audio
        filename = tmp.name
    await communicate.save(filename)  # Save the generated speech to the temporary file
    return filename  # Return the file path of the saved audio

# Function to play the generated speech
async def speak_text(bot_response):
    global interrupt_event
    print(f"[DEBUG] TTS input: {repr(bot_response)}")  # Print the bot's response for debugging
    filename = await get_speech(bot_response)  # Get the speech audio file for the response
    if audio_available:  # Check if audio is available
        pygame.mixer.music.load(filename)  # Load the audio file
        pygame.mixer.music.play()  # Play the audio
        while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
            if interrupt_event:  # If an interrupt event occurs (e.g., close or stop music)
                pygame.mixer.music.stop()  # Stop the audio
                break
            pygame.time.Clock().tick(10)  # Wait for 10 milliseconds to avoid locking up the system

        if interrupt_event:  # If there was an interruption, restart the audio playback
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
            pygame.time.Clock().tick(10)        
        os.remove(filename)  # Delete the temporary audio file after playing

# ────────────────────────────────────────────────
# Speech Recognition
# ────────────────────────────────────────────────

# Function to listen for and transcribe user input
def listen_to_user():
    with sr.Microphone() as source:  # Use the microphone as input source
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise
        print("Listening...")  # Print status for debugging
        try:
            audio = recognizer.listen(source, timeout=7)  # Listen for audio input, with a 7-second timeout
            text = recognizer.recognize_google(audio).lower()  # Recognize speech and convert to lowercase
            print(f"You said: {text}")  # Print the transcribed text for debugging
            return text  # Return the transcribed text
        except:  # If there's an error (e.g., timeout or no speech detected)
            return None  # Return None

# Function to listen for and capture the user's name
def name_listener():
    with sr.Microphone() as source:  # Use the microphone as input source
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise
        print("Say your name...")  # Prompt user to say their name
        try:
            audio = recognizer.listen(source, timeout=7)  # Listen for audio input, with a 7-second timeout
            return recognizer.recognize_google(audio).lower(), True  # Return the recognized name and validity
        except:  # If there's an error (e.g., timeout or no speech detected)
            return None, False  # Return None and False to indicate failure

# ────────────────────────────────────────────────
# Response Generator
# ────────────────────────────────────────────────

# Function to generate a response based on user input
def generate_response(user_input):
    user_input = user_input.lower()  # Convert user input to lowercase for consistent matching
    jokes = [  # A list of jokes to respond with
        "Why don't skeletons fight each other? They don't have the guts.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Why don’t oysters donate to charity? Because they are shellfish.",
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "What did the ocean say to the beach? Nothing, it just waved."
    ]
    
    # Various checks to determine the response based on user input
    if "who are you" in user_input:
        return "I am Wonder, your personal AI Assistant"
    elif "a joke" in user_input:
        return random.choice(jokes)  # Return a random joke
    elif "hey wonder" in user_input:
        return "Yes?"  # Respond to "Hey Wonder"
    elif "change my name" in user_input:
        return "name has been updated"  # Indicate that the name has been updated
    elif "exit" in user_input:
        return f"Goodbye {name_of_user}! I will shut off now."  # Exit message
    elif "detect" in user_input:
        return "Detecting"  # Trigger object detection
    elif "help" in user_input:
        return ("Say 'who are you' to get a description of this device. \n"
                "Say 'a joke' to get a joke. \n"
                "Say 'change my name' to change the name we refer to you as. \n"
                "Say 'detect' to see what's in front of you. \n"
                "Say 'exit' to stop running the device.")  # Help message with available commands
    else:
        # Chatbot logic using the DialoGPT model
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_output = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, 
                                    no_repeat_ngram_size=3, top_p=0.92, top_k=50, temperature=0.5)
        bot_response = tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {bot_response}")  # Print chatbot response for debugging
        return bot_response  # Return the generated chatbot response

# ────────────────────────────────────────────────
# Disparity to Distance
# ────────────────────────────────────────────────

# Function to convert disparity values to distance in inches
def disparity_to_distance(disparities):
    valid_disparities = [d for d in disparities if d > 0]  # Filter out invalid disparity values (0)
    if not valid_disparities:  # If no valid disparity values are found
        return float('inf')  # Return infinity (no object detected)
    avg_disparity = np.mean(valid_disparities)  # Calculate the average disparity
    distance = 140.99287 - 28.55858 * np.log(avg_disparity)  # Convert disparity to distance using a formula
    return round(distance, 2)  # Return the distance rounded to 2 decimal places

# ────────────────────────────────────────────────
# Camera Thread with Object Detection and Distance Calculation
# ────────────────────────────────────────────────

# Function to handle the camera stream and object detection
def camera_thread():
    global interrupt_event, detecting  # Declare global variables
    pipeline = dai.Pipeline()  # Create a DepthAI pipeline

    # Setup camera nodes and streams (left/right mono cameras, stereo depth, etc.)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    manip = pipeline.create(dai.node.ImageManip)
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    # Camera setup
    monoLeft.setCamera("left")
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setCamera("right")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    manip.initialConfig.setResize(300, 300)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

    nn.setBlobPath("mobilenet-ssd_openvino_2021.4_6shave.blob")
    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)

    nnOut.setStreamName("nn")
    disparityOut.setStreamName("disparity")
    xoutRight.setStreamName("rectifiedRight")

    # Linking nodes
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.rectifiedRight.link(manip.inputImage)
    stereo.disparity.link(disparityOut.input)
    manip.out.link(xoutRight.input)
    manip.out.link(nn.input)
    nn.out.link(nnOut.input)

    with dai.Device(pipeline) as device:  # Start the device with the pipeline
        qRight = device.getOutputQueue("rectifiedRight", 4, False)
        qDisparity = device.getOutputQueue("disparity", 4, False)
        qDet = device.getOutputQueue("nn", 4, False)
        disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

        while True:  # Main loop for capturing frames and performing detection
            rightFrame = qRight.get().getCvFrame() if qRight.has() else None
            disparityFrame = qDisparity.get().getFrame() if qDisparity.has() else None
            detections = qDet.get().detections if qDet.has() else []

            closest_distance = float('inf')  # Initialize the closest distance as infinite
            closest_label = None  # Initialize the closest label as None
            cx = 0  # Initialize center x coordinate

            if rightFrame is not None and disparityFrame is not None:
                # Normalize disparity frame for visualization
                disparityFrameNorm = (disparityFrame * disparityMultiplier).astype(np.uint8)
                disparityColor = cv2.applyColorMap(disparityFrameNorm, cv2.COLORMAP_JET)

                # Process each detected object
                for detection in detections:
                    label = label_map[detection.label]
                    x1 = int(detection.xmin * rightFrame.shape[1])
                    y1 = int(detection.ymin * rightFrame.shape[0])
                    x2 = int(detection.xmax * rightFrame.shape[1])
                    y2 = int(detection.ymax * rightFrame.shape[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Calculate distance based on disparity at the object's center
                    if 0 <= cx < disparityFrame.shape[1] and 0 <= cy < disparityFrame.shape[0]:
                        points = [
                            disparityFrame[cy, cx],
                            disparityFrame[y1, x1] if y1 < disparityFrame.shape[0] and x1 < disparityFrame.shape[1] else 0,
                            disparityFrame[y1, x2] if y1 < disparityFrame.shape[0] and x2 < disparityFrame.shape[1] else 0,
                            disparityFrame[y2, x1] if y2 < disparityFrame.shape[0] and x1 < disparityFrame.shape[1] else 0,
                            disparityFrame[y2, x2] if y2 < disparityFrame.shape[0] and x2 < disparityFrame.shape[1] else 0
                        ]
                        distance_in = disparity_to_distance(points)
                        if distance_in < closest_distance:
                            closest_distance = round(distance_in)
                            closest_label = label

                # Check if the detected object is too close
                if closest_label and closest_distance < 12:
                    interrupt_event = True
                    detecting = f"{closest_label} is {closest_distance} inches in front"
                    asyncio.run(speak_text(detecting))  # Announce detection
                    time.sleep(1)  # Delay before continuing
                    interrupt_event = False

                # Display frames
                cv2.imshow("Rectified Right", rightFrame)
                cv2.imshow("Disparity", disparityColor)

            # Quit loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()  # Close the windows once finished

# ────────────────────────────────────────────────
# Chat Thread
# ────────────────────────────────────────────────

# Asynchronous function for handling user commands
async def chat_thread():
    global name_of_user
    while True:
        await speak_text("Say your name please")  # Ask for user name
        name_of_user, valid = name_listener()  # Listen for user's name
        if valid:
            await speak_text(f"Hey {name_of_user}")  # Greet the user with their name
            break

    while True:
        await speak_text("Ready for command")  # Wait for user command
        user_input = listen_to_user()  # Listen for user input
        if not user_input:
            await speak_text("Say 'help' if you need the list of commands you can say")  # If no input, ask for help
            continue
        print(f"{name_of_user} said: {user_input}")  # Print user input for debugging
        response = generate_response(user_input)  # Generate a response to the user input
        if "name has been updated" in response:
            await speak_text("Say your new name")  # Ask for the new name if changed
            while True:
                name_of_user, valid = name_listener()  # Listen for new name
                if valid:
                    await speak_text(f"Hey {name_of_user}")  # Greet the user with their new name
                    break
        elif "detect" in user_input:
            await speak_text(detecting if detecting else "Nothing close detected yet.")  # Announce object detection status            
        else:
            await speak_text(f"{response}")  # Speak out the bot's response
        if "exit" in user_input:
            cv2.destroyAllWindows()  # Close any open windows
            os._exit(0)  # Exit the program

# ────────────────────────────────────────────────
# Launch Threads
# ────────────────────────────────────────────────

# Main entry point for launching the assistant
if __name__ == "__main__":
    asyncio.run(speak_text("Hello! I am Wonder, your personal AI assistant. I can help you with various tasks. Please wait 5 seconds for devices to pair"))  # Greet the user and notify about device pairing
    cam_thread = Thread(target=camera_thread, daemon=True)  # Start the camera thread
    cam_thread.start()

    print("⏳ Waiting 5 seconds for devices to pair...")  # Print pairing message
    for i in range(5):
        time.sleep(1)
        print(f"Startup: {i + 1}s")

    asyncio.run(chat_thread())  # Start the chat thread to handle user input and interaction

