import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from utlis import *


def load_training_metadata(driverfile):
    video_files = []
    transcriptions = []
    with open(driverfile, 'r') as f:
        for line in f:
            video_file, transcription = line.strip().split('|')
            video_files.append(video_file)
            transcriptions.append(transcription)
    return video_files, transcriptions
def create_lip_reading_model(input_shape, num_classes):
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'), input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(256, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
# Load frames, audio, and text content
def load_data(frames_path, audio_path, text_content):
    # Load frames (JPEG files)
    frames = []  # List to store frame data
    # Iterate through frames_path and load each frame, preprocess if necessary
    # Append each frame to the frames list
    # Load audio
    audio_data, sampling_rate = load_audio(audio_path)
    # Preprocess text content
    # Convert text content into numerical format (e.g., one-hot encoding)
    return np.array(frames), audio_data, text_content

# Load audio
def load_audio(audio_path):
    # Load audio file using a suitable library (e.g., librosa)
    # Preprocess audio data if necessary (e.g., convert to spectrogram)
    audio_data, sampling_rate = None, None  # Placeholder, replace with actual audio data and sampling rate
    return audio_data, sampling_rate

# Train the lip reading model
def train_lip_reading_model(frames, audio_data, text_content):
    model = create_lip_reading_model()
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit([frames, audio_data], text_content, epochs=10, validation_split=0.2)

    return model

videos_directory = "path_to_videos_directory"
video_files, transcriptions = load_training_metadata('lrwinputlist_1.txt')
print("Number for videos for training: " + str(len(video_files)))
print("Number for transcriptions for training: " + str(len(transcriptions)))


# Loop through videos
for video_folder in os.listdir(videos_directory):
    video_folder_path = os.path.join(videos_directory, video_folder)
    frames_path = os.path.join(video_folder_path, "frames")
    audio_path = os.path.join(video_folder_path, "audio.wav")
    text_content_path = os.path.join(video_folder_path, "text_content.txt")

    # Load text content
    with open(text_content_path, "r") as file:
        text_content = file.readlines()

    # Load data
    frames, audio_data, text_content = load_data(frames_path, audio_path, text_content)

    # Train lip reading model
    lip_reading_model = train_lip_reading_model(frames, audio_data, text_content)

    # Save the trained model
    model_filename = f"{video_folder}_lip_reading_model.h5"
    lip_reading_model.save(model_filename)
    print(f"Model trained and saved for video {video_folder}.")
