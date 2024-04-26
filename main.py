import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
import cv2
import mediapipe as mp
import time
import math
import threading
import queue
# import pyttsx3
import pickle
import numpy as np

# # Initialize TTS engine
# engine = pyttsx3.init()
# engine.setProperty('volume', 0.9)
# engine.setProperty('rate', 150)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

# Speech command queue
speech_queue = queue.Queue()

# Thread safe lock
lock = threading.Lock()

# def speech_processor():
#     while True:
#         item = speech_queue.get()
#         text = item['text']
#         repeat = item.get('repeat', 1)
#         delay = item.get('delay', 0)
        
#         for _ in range(repeat):
#             engine.say(text)
#             engine.runAndWait()
#             if _ < repeat - 1:
#                 time.sleep(delay)
        
#         speech_queue.task_done()

# def speak(text, repeat=1, delay=0):
#     speech_queue.put({'text': text, 'repeat': repeat, 'delay': delay})

# threading.Thread(target=speech_processor, daemon=True).start()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # img = self.undistort_frame(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            # Define reference landmarks
            left_shoulder = landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value]

            # Calculate shoulder width
            shoulder_distance_pixels = self.calculate_distance(
                (left_shoulder.x * img.shape[1], left_shoulder.y * img.shape[0]),
                (right_shoulder.x * img.shape[1], right_shoulder.y * img.shape[0])
            )
            # Conversion factor (example, should be adjusted or calculated based on calibration)
            conversion_factor = 0.0254  # Inches per pixel (example value)
            shoulder_width_inches = shoulder_distance_pixels * conversion_factor

            # Display measurement on the frame
            cv2.putText(img, f"Shoulder Width: {shoulder_width_inches:.2f} inches", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format='bgr24')

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# WebRTC configuration (optional, for specific network scenarios)
rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:your_turn_server_here"], "username": "your_username", "credential": "your_password"}
    ]
})


webrtc_streamer(key="example", video_processor_factory=VideoProcessor, media_stream_constraints={
        'video': {
            'width': 1920
            },
        'audio': False
    },
     rtc_configuration=rtc_config)

