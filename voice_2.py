import openai
import pyttsx3
import os
import sounddevice as sd
import numpy as np
import wave
import pvporcupine
import pyaudio
import time
import struct
from camera_detect import camera_detect
from arm_move import MoveArm
from pymycobot import *
import threading
import serial.tools.list_ports
from dotenv import load_dotenv

load_dotenv()

def find_arduino_port():
    print("here")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check if the port description or device name contains typical Arduino indicators
        if "Arduino" in port.description or "usbmodem" in port.device:
            return port.device
    return None

arduino_port = find_arduino_port()

if arduino_port:
    print("Arduino found at:", arduino_port)
    mc = MyCobot280(arduino_port, 115200)  # 设置端口
else:
    print("Arduino not found!")

# Initialize pyttsx3 engine (fallback if needed)
engine = pyttsx3.init()
camera_params = np.load("camera_params.npz") 
mtx, dist = camera_params["mtx"], camera_params["dist"]
m = camera_detect(0, 43, mtx, dist,mc)
moveArm = MoveArm(mc) 
mc.set_vision_mode(1)
# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global conversation history for context (only for general queries)
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant. Please provide short, concise, and direct answers."}
]

# Function to recorßd audio from microphone
def record_audio(filename="speech.wav", duration=4, samplerate=44100):
    print("Recording... Speak now.")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait for recording to finish

    # Save as a WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print("Recording complete.")
    return filename

def transcribe_audio(filename="speech.wav"):
    """Transcribes audio using OpenAI Whisper API."""
    with open(filename, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
        )
    return response.text

def speak(text, filename="response.mp3"):
    """Converts text to speech using OpenAI TTS API."""
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    with open(filename, "wb") as audio_file:
        audio_file.write(response.content)  # Save the generated speech
    
    # Play the audio file (Mac/Linux: afplay, Windows: start)
    os.system(f"afplay {filename}" if os.name == "posix" else f"start {filename}")
    print("Speaking:", text)

def ask_openai(query):
    """Uses OpenAI GPT-4 to answer general queries with context retention."""
    # Append the new query to the conversation history
    conversation_history.append({"role": "user", "content": query})
    
    # Set a default token limit; increase if query seems to require a longer answer
    max_tokens_val = 50
    if any(keyword in query.lower() for keyword in ["explain", "describe", "theorem", "how", "why"]):
         max_tokens_val = 150  # adjust as needed

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        max_tokens=max_tokens_val
    )
    answer = response.choices[0].message.content.strip()
    
    # Append the assistant's answer to the conversation history
    conversation_history.append({"role": "assistant", "content": answer})
    return answer


def get_intent_from_openai(command):
    """
    Uses OpenAI GPT to classify intent based on a user command.
    For "pick_object" intents, it also identifies the object and any directional instruction.
    If the command only indicates a basic pickup action (e.g. "pick up" or "pickup"), then it means pick it up and return to the original position.
    If the command includes "move left" or "move right", return the corresponding direction.
    If the command includes "move to me" or "bring to me", return that instruction.
    The returned list for a "pick_object" intent should have a minimum of 2 elements (intent and object)
    and a maximum of 3 elements (intent, object, and direction).
    """
    prompt = f"""
    The user has given a voice command: "{command}".
    Identify if the command is related to MyCobot actions. If so, return one of the following intents:
    - "pick_object"
    - "drop_object"
    - "move_arm"
    - "stop"
    - "unknown" (if it's not related to the robot)

    If the intent is "pick_object", identify the object in the command and return the object name. The object must be one of the following:
    0: 'person', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 39: 'bottle', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 58: 'potted plant', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors'

    Additionally, for "pick_object" intents:
    - If the command only indicates a pickup action (e.g. "pick up" or "pickup") without any directional words, interpret it as picking up the object and returning to the original position.
      In this case, return a list with three elements: ["pick_object", object name, pick].
    - If the command includes "move left", then include "left" as a third element.
    - If the command includes "move right", then include "right" as a third element.
    - If the command includes "move to me" or "bring to me", then include "to_me" as a third element.

    If the intent is "move_arm", return the intent and the direction (e.g: "home", "me", "observe", "origin")

    Respond with only a Python list containing the intent keyword and, if applicable, the object name and direction. Nothing else.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    
    try:
        intent_info = eval(answer)
        if not isinstance(intent_info, list):
            intent_info = [intent_info]
    except Exception as e:
        intent_info = [answer]
    return intent_info


def execute_robot_command(intent_info):
    """
    Executes MyCobot actions based on the intent classification.
    """
    if isinstance(intent_info, list):
        intent = intent_info[0]
        obj = intent_info[1] if len(intent_info) > 1 else None
        print("obj",obj)
        direction = intent_info[2] if len(intent_info) > 2 else None
    else:
        intent = intent_info
        obj = None

    if intent == "pick_object":
        supported_objects = [
            'person', 'backpack', 'umbrella', 'handbag', 'bottle', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'potted plant', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors'
        ]
        objects = {'person': 0, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'bottle': 39, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'potted plant': 58, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76}
        if obj is None:
            response_text = "No object specified for picking up."
        elif obj.lower() not in [s.lower() for s in supported_objects]:
            response_text = f"Sorry, I cannot pick up '{obj}' because it is not supported."
        else:
            response_text = f"I am picking up the {obj}."
            thread_speak = threading.Thread(target=speak, args=(f"I am picking up the {obj}.",))
            thread_trace = threading.Thread(target=m.vision_trace_loop_yolo, args=(mc,objects[obj],direction))

            # Start both threads
            thread_speak.start()
            thread_trace.start()

        print(response_text)
        return response_text

    elif intent == "track_face":
        print("Activating face tracking.")
        return "Activating face tracking."
    elif intent == "drop_object": 
        print("Dropping the object.")
        response_text = f"Dropping the object."
        thread_speak = threading.Thread(target=speak, args=(response_text,))
        thread_drop = threading.Thread(target=m.drop())
        thread_speak.start()
        thread_drop.start()
        # m.drop()
        return "Dropping the object."
    elif intent == "move_arm":
        response_text = f"Moving MyCobot to the {obj} position."
        thread_speak = threading.Thread(target=speak, args=(response_text,))
        thread_move = threading.Thread(target=moveArm.move_to_position, args=(obj,))
        thread_speak.start()
        thread_move.start()
        return response_text

    elif intent == "stop":
        print("Stopping MyCobot.")
        return "Stopping MyCobot."

    # For "unknown" or unrecognized intents, fall back to a general query
    return None

def process_command(command):
    # Get intent classification from OpenAI (as a list)
    intent_info = get_intent_from_openai(command)
    
    # If the command matches a MyCobot action, execute it
    robot_response = execute_robot_command(intent_info)
    if robot_response:
        return (False,robot_response)
    
    # Otherwise, process as a general question using GPT-4 with context
    return (True,ask_openai(command))

# Wake word detection using Porcupine with the "jarvis" keyword
def porcupine_wake_word_detection():
    porcupine = pvporcupine.create(
        access_key=os.getenv("PORCUPINE_ACCESS_KEY"),
        keyword_paths=["/Users/devanggupta/Library/CloudStorage/OneDrive-Personal/Desktop/CityU/FYP/Hey-Nova_en_mac_v3_0_0.ppn"]
    )
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length)
    print("Listening for the wake word...")


    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            result = porcupine.process(pcm)
            if result >= 0:
                print("Wake word detected!")
                # Once detected, record and process the user's command
                audio_file = record_audio()  # Record user command
                command = transcribe_audio(audio_file)  # Convert speech to text
                if command:
                    print("User:", command)
                    flag,response = process_command(command)
                    print("Assistant:", response)
                    speak(response) if flag == True else None
    except KeyboardInterrupt:
        print("Stopping wake word detection.")
    finally:
        stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    porcupine_wake_word_detection()
