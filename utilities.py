from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import os
import librosa
import cv2
import numpy as np


def load_training_metadata(driverfile):
    video_files = []
    transcriptions = []
    with open(driverfile, 'r') as f:
        for line in f:
            video_file, transcription = line.strip().split('|')
            video_files.append(video_file)
            transcriptions.append(transcription)
    return video_files, transcriptions

def get_video_frames_np(framesDirectory):

    files = os.listdir(framesDirectory)

    # Filter out only the .png files
    png_files = [file for file in files if file.endswith('.png')]
    frame_count = len(png_files)
    processed_frames = []
    if frame_count > 0:
        for i in range(frame_count):
            frame_path = os.path.join(framesDirectory, f'frame{i}.png')
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
            frame = frame.astype(np.float32) / 255.0  # Normalize pixel values
            processed_frames.append(frame)

    return np.array(processed_frames)

def get_audio(audioFilePath):
    audio, sr = librosa.load(audioFilePath, sr=16000)
    return audio