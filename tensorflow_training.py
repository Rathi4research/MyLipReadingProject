import os
import numpy as np
import tensorflow as tf
from keras import layers, models, Model
# from utlis import *
from utilities import *

# def create_lip_reading_model(input_shape, num_classes):
#     model = models.Sequential([
#         layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'), input_shape=input_shape),
#         layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
#         layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
#         layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
#         layers.TimeDistributed(layers.Conv2D(256, (3, 3), activation='relu')),
#         layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
#         layers.TimeDistributed(layers.Flatten()),
#         layers.LSTM(256, return_sequences=True),
#         layers.LSTM(128),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

def create_lip_reading_model(num_classes):

    # Define the model using the Sequential API
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'), input_shape=(None, 224, 224, 3)),
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
    # video_input = layers.Input(shape=(None, 224, 224, 3))  # Variable-length sequence of video frames
    # # Define audio input layer
    # audio_input = layers.Input(shape=(None,))  # Variable-length sequence of audio samples
    #
    # # Video feature extraction using CNN (e.g., ResNet)
    # cnn_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    # cnn_output = layers.TimeDistributed(cnn_base)(video_input)
    # cnn_output = layers.GlobalAveragePooling2D()(cnn_output)
    #
    # # LSTM layer for temporal processing of video features
    # video_lstm = layers.LSTM(64)(cnn_output)
    # # LSTM layer for temporal processing of audio features
    # audio_lstm = layers.LSTM(64)(audio_input)
    # # Concatenate LSTM outputs
    # combined_features = layers.Concatenate()([video_lstm, audio_lstm])
    #
    # # Additional layers if needed
    # # For example:
    # # combined_features = layers.Dense(128, activation='relu')(combined_features)
    # # Output layer for text transcription
    # output = layers.Dense(num_classes, activation='softmax')(combined_features)
    #
    # # Create model
    # model = Model(inputs=[video_input, audio_input], outputs=output)
    # return model

# Train the lip reading model
def train_lip_reading_model(frames, audio_data, text_content):
    model = create_lip_reading_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit([frames, audio_data], text_content, epochs=10, validation_split=0.2)
    return model

video_files, transcriptions = load_training_metadata('lrwinputlist_mac.txt')
print("Number for videos for training: " + str(len(video_files)))
print("Number for transcriptions for training: " + str(len(transcriptions)))


# Loop through videos
text_labels_count = len(transcriptions)
text_counter = 0
videos_list = []
audios_list = []

for video_folder in video_files:
    frames_path = os.path.join(video_folder, "frames")
    index_folder = os.path.basename(os.path.normpath(video_folder))
    audio_path = os.path.join(video_folder, "audio",index_folder + '.wav')

    videos_list.append(get_video_frames_np(frames_path))
    audios_list.append(get_audio(audio_path))

labels = np.array(transcriptions)
videos = np.array(videos_list)
audios = np.array(audios_list)
model = create_lip_reading_model(11)

print("Data Types:")
print("Videos:", type(videos))
print("Audios:", type(audios))
print("Labels:", type(labels))

exit(0)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit([videos, audios], labels, batch_size=32, epochs=10, validation_split=0.2)
# Save the trained model to a file
model.save("lip_reading_model.h5")
