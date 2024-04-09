import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from utlis import *

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
    frames = get_video_frames(frames_path)  # List to store frame data
    # Load audio
    audio_data = get_audio(audio_path)
    return np.array(frames), audio_data, text_content


# Train the lip reading model
def train_lip_reading_model(frames, audio_data, text_content):

    model = create_lip_reading_model()
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit([frames, audio_data], text_content, epochs=10, validation_split=0.2)

    return model

video_files, transcriptions = load_training_metadata('lrwinputlist_1.txt')
print("Number for videos for training: " + str(len(video_files)))
print("Number for transcriptions for training: " + str(len(transcriptions)))


# Loop through videos
text_labels_count = len(transcriptions)
text_counter = 0
for video_folder in video_files:
    frames_path = os.path.join(video_folder, "frames")
    index_folder = os.path.basename(os.path.normpath(video_folder))
    audio_path = os.path.join(video_folder, "audio",index_folder + '.wav')
    video_text = transcriptions[text_counter]
    text_array = []
    text_array.append(video_text)
    text_content = preprocess_text(text_array)
    text_counter += 1
    # Load data

    frames, audio_data, text_content = load_data(frames_path, audio_path, text_content)

    # Train lip reading model
    lip_reading_model = train_lip_reading_model(frames, audio_data, text_content)

    # Save the trained model
    model_filename = "lip_reading_model.h5"
    lip_reading_model.save(model_filename)
    print(f"Model trained and saved for video {video_folder}.")
