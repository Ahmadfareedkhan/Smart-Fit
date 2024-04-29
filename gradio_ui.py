import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import sys

import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



class VideoProcessor:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def process_video(self, input_video_path):
        cap = cv2.VideoCapture(input_video_path)
        output_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(imgRGB)

            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # Additional processing can be added here
            output_frames.append(frame)

        cap.release()
        output_video_path = 'output.mp4'
        self.save_video(output_frames, output_video_path)
        return output_video_path

    def save_video(self, frames, output_path):
        height, width, layers = frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

        for frame in frames:
            out.write(frame)
        out.release()

def process_and_display(input_video):
    processor = VideoProcessor()
    output_video = processor.process_video(input_video)
    return output_video

iface = gr.Interface(
    fn=process_and_display,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.Video(label="Processed Video"),
    title="Pose Estimation Video Processor"
)

iface.launch()
