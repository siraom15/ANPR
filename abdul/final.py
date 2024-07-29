import speech_recognition as sr
import cv2
import os
from datetime import datetime
import time
import openai
import base64
import requests
from gtts import gTTS
import pygame
import re
from tkinter import *
import customtkinter
from PIL import Image, ImageTk
import numpy as np
import threading

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()
api_key = ""
history_file = "history.txt"
sound_dir = "sounds"

# List of video files
video_files = ['abdul.mp4']
current_video_index = 0

# Define a video capture object for the first video file
abdul = cv2.VideoCapture(video_files[current_video_index])
cam = cv2.VideoCapture(0)  # Second camera

# Declare the width and height in variables
width, height = 200, 200

# Set the width and height for both video and camera
abdul.set(cv2.CAP_PROP_FRAME_WIDTH, width)
abdul.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create a GUI app
# app = Tk()
app = customtkinter.CTk()
app.title('Abdul')
customtkinter.set_appearance_mode("dark")

# Bind the app with Escape keyboard to quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

# Create labels and display them on app
label_widget1 = Label(app)
label_widget2 = Label(app)
label_widget1.grid(row=0, column=1)
label_widget2.grid(row=0, column=0)

history_label = customtkinter.CTkLabel(app, text="ประวัติ")
history_label.grid(row=6, column=0, columnspan=2)

def remove_green_screen(frame):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range of green color in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask to extract the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert the mask to get everything except the green color
    mask_inv = cv2.bitwise_not(mask)
    
    # Convert the frame to RGBA to add transparency
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
    # Apply the mask to the frame
    frame_rgba[:, :, 3] = mask_inv
    
    return frame_rgba

def open_abdul():
    global current_video_index, abdul

    ret, frame = abdul.read()
    if not ret:
        current_video_index = (current_video_index + 1) % len(video_files)
        abdul = cv2.VideoCapture(video_files[current_video_index])
        ret, frame = abdul.read()

    frame = remove_green_screen(frame)
    captured_image = Image.fromarray(frame)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget1.photo_image = photo_image
    label_widget1.configure(image=photo_image)
    label_widget1.after(10, open_abdul)

def open_camera():
    ret, frame = cam.read()
    if not ret:
        return
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget2.photo_image = photo_image
    label_widget2.configure(image=photo_image)
    label_widget2.after(10, open_camera)

def initialize_microphone():
    print(sr.Microphone.list_microphone_names())
    return sr.Microphone(0)

def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        label_abdul_talk.configure(text = "กำลังฟัง...")
        audio = recognizer.listen(source, phrase_time_limit=3)
        
    response = {"success": True, "error": None, "transcription": None}

    try:
        res = recognizer.recognize_google(audio, language="th-TH")
        response["transcription"] = res
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    
    return response

def capture_image():
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(output_dir, f"image_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image saved at {image_path}")
    
    return image_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_path):
    base64_image = encode_image(image_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s this main object? Main object is something holded in hand. Please answer in Thai language in this template. Template: สิ่งของนี้คือ ... ภาษาอังกฤษคือ ... ภาษาจีนคือ ..."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    
    if response.status_code == 200:
        choices = response_json.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            content = message.get('content', '')
            print("Response content:", content)
            label_abdul_talk.configure(text = content)
            Thai, Eng, Chinese = extract_thai_eng_chinese(content)
            save_to_history(image_path, Thai, Eng, Chinese)
            play_text_to_speech(Thai, 'th', 'TH')
            play_text_to_speech(Eng, 'en', 'EN')
            play_text_to_speech(Chinese, 'zh', 'CH')
        else:
            print("No choices found in the response.")
    else:
        print("Request failed with status code:", response.status_code)
        print("Response:", response_json)

def extract_thai_eng_chinese(content):
    match = re.match(r"(.*ภาษาอังกฤษคือ)(.*?)(ภาษาจีนคือ)(.*)", content)
    if match:
        Thai = match.group(1).strip()
        Eng = match.group(2).strip()
        Chinese = match.group(4).strip()
        print("Thai:", Thai)
        print("Eng:", Eng)
        print("Chinese:", Chinese)
        return Thai, Eng, Chinese
    else:
        print("No match found.")
        return "", "", ""
    
def play_text_to_speech(text, lang, name):
    tts = gTTS(text=text, lang=lang)
    filename = f"output-{name}.mp3"
    tts.save(filename)
    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)

def save_to_history(image_path, thai, eng, chinese):
    with open(history_file, "a", encoding='utf-8') as file:  # Specify UTF-8 encoding
        file.write(f"{image_path}|{thai}|{eng}|{chinese}\n")
    update_history_display()

def show_popup(image_path, thai, eng, chinese):
    popup = Toplevel(app)
    popup.title("Image Preview")
    
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.LANCZOS)  # Resize image for the popup window
    img = ImageTk.PhotoImage(img)
    
    img_label = Label(popup, image=img)
    img_label.image = img  # Keep a reference to avoid garbage collection
    img_label.pack()
    
    text = f"Thai: {thai}\nEng: {eng}\nChinese: {chinese}"
    text_label = Label(popup, text=text)
    text_label.pack()

def update_history_display():
    if os.path.exists(history_file):
        with open(history_file, "r", encoding='utf-8') as file:  # Specify UTF-8 encoding
            lines = file.readlines()
        
        lines = lines[-9:][::-1]  # Get the last 10 entries and reverse the order
        
        # Clear previous widgets
        for widget in app.grid_slaves(row=7):
            widget.grid_forget()
        for widget in app.grid_slaves(row=8):
            widget.grid_forget()
        for widget in app.grid_slaves(row=9):
            widget.grid_forget()
        
        for i, line in enumerate(lines):
            image_path, thai, eng, chinese = line.strip().split('|')
            img = Image.open(image_path)
            img = img.resize((100, 100), Image.LANCZOS)  # Resize image to fit in the GUI
            img = ImageTk.PhotoImage(img)
            
            img_label = Label(app, image=img)
            img_label.image = img  # Keep a reference to avoid garbage collection
            img_label.bind("<Button-1>", lambda e, path=image_path, t=thai, en=eng, ch=chinese: show_popup(path, t, en, ch))
            row, col = divmod(i, 3)
            img_label.grid(row=7 + row*2, column=col)

            text = f"Thai: {thai}\nEng: {eng}\nChinese: {chinese}"
            text_label = customtkinter.CTkLabel(app, text=text)
            text_label.grid(row=9 + row*2, column=col)

def analyst_image():
    label_abdul_talk.configure(text = "กำลังวิเคราะห์รูปภาพ...")
    play_text_to_speech('กำลังวิเคราะห์รูปภาพ', 'th', 'analysing')
    image_path = capture_image()
    if image_path:
        describe_image(image_path)

def start_abdul_voice_control():
    label_abdul_talk.configure(text = "อับดุล: สวัสดีฉันคืออับดุล ถามฉันได้เลยค่ะ")
    label_you_talk.configure(text = "คุณ: เปิดใช้งานการสั่งการด้วยเสียงแล้ว")
    play_text_to_speech('สวัสดีฉันคืออับดุล ถามฉันได้เลยค่ะ', 'th', 'start')
    mic = initialize_microphone()
    
    while True:
        print("Say something!")
        speech = recognize_speech_from_mic(recognizer, mic)
        
        if speech["transcription"]:
            print("You said: {}".format(speech["transcription"]))
            label_you_talk.configure(text = "คุณพูด: "+ speech["transcription"])

            if "อันนี้คืออะไร" in speech["transcription"]:
                analyst_image()
            if "ออกจากโปรแกรม" in speech["transcription"]:
                stop_abdul_voice_control()
                exit(0) #Exit Thread

        if not speech["success"]:
            label_abdul_talk.configure(text = "อับดุล: ฉันไม่เข้าใจที่คุณพูด ลองพูดใหม่อีกรอบ")
            # play_text_to_speech('ฉันไม่เข้าใจที่คุณพูด ลองพูดใหม่อีกรอบ', 'th', 'again')
        
        if speech["error"]:
            label_abdul_talk.configure(text = "อับดุล: ฉันไม่เข้าใจที่คุณพูด ลองพูดใหม่อีกรอบ")
            # play_text_to_speech('ฉันไม่เข้าใจที่คุณพูด ลองพูดใหม่อีกรอบ', 'th', 'again')


def stop_abdul_voice_control():
    label_abdul_talk.configure(text = "อับดุล: สวัสดี")
    label_you_talk.configure(text = "คุณ: ยังไม่ได้เปิดการใช้งานด้วยเสียง")
    play_text_to_speech('ปิดการใช้งานสั่งการด้วยเสียงแล้ว', 'th', 'stop')

# Start the video feeds and speech recognition
open_abdul()
open_camera()

def exit_abdul():
    exit(0)

def clear_history():
    with open(history_file, "w", encoding='utf-8') as file:
        pass  # Specify UTF-8 encoding
    update_history_display()
    return

label_abdul_talk = customtkinter.CTkLabel(app, text="อับดุล: สวัสดี")
label_abdul_talk.grid(row=2, column=1)

label_you_talk = customtkinter.CTkLabel(app, text="คุณ: ยังไม่ได้เปิดการใช้งานด้วยเสียง")
label_you_talk.grid(row=2, column=0)

button_start_abdul = customtkinter.CTkButton(app, text="เริ่มต้นใช้เสียงสั่งการ", command=lambda: threading.Thread(target=start_abdul_voice_control).start())
button_start_abdul.grid(row=3, column=0)
button_analyst_image = customtkinter.CTkButton(app, text="อันนี้คืออะไร", command=lambda: threading.Thread(target=analyst_image).start())
button_analyst_image.grid(row=4, column=0)
exit_btn = customtkinter.CTkButton(app, text="ล้างประวัติ", command=clear_history)
exit_btn.grid(row=5, column=0)
exit_btn = customtkinter.CTkButton(app, text="ปิดโปรแกรม", command=exit_abdul)
exit_btn.grid(row=5, column=1)

label_hint = customtkinter.CTkLabel(app, text="คำสั่งเสียง: อันนี้คืออะไร, ออกจากโปรแกรม")
label_hint.grid(row=1, column=3)
# Initialize the history display
update_history_display()

# Create an infinite loop for displaying app on screen
app.mainloop()
