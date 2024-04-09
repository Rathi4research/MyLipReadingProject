import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
import torch.optim as optim
import ffmpeg
import os
import librosa
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer



def extract_frames_from_video(video_path,frames_dir, needwrite=False):
    os.makedirs(frames_dir, exist_ok=True)
    # Capture video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    # Read frames and save as images
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f'frame{frame_count}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return frame_count

def extract_audio_from_video(video_path,audio_dir,audio_filename, needwrite=False):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_file_path = os.path.join(audio_dir,audio_filename)
    audio.write_audiofile(audio_file_path)
    audio, sr = librosa.load(audio_file_path, sr=16000)  # Adjust sample rate as needed

def preprocess_video(video_path):
    directory, file = os.path.split(video_path)
    folder_name = os.path.basename(directory)
    file_name = str(file).replace('.mp4','')
    training_root_dir = 'D:\Codebase\lip_reading_project_public\sentence_lip_reading\lrs2_training_video_dir'
    # Use ffmpeg to extract frames from the video
    print("Extracting frames from the video: " + video_path)
    frames_dir = os.path.join(training_root_dir,folder_name,file_name,'frames')
    audio_dir = os.path.join(training_root_dir,folder_name,file_name,'audio')
    audio_filename = file_name + '.wav'
    print(f'Writing frames to the directory %s', frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    # Read frames and save as images
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f'frame{frame_count}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    print("Frames directory: " + frames_dir)

    processed_frames = []
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f'frame{i}.png')
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
        frame = frame.astype(np.float32) / 255.0  # Normalize pixel values
        processed_frames.append(frame)

    return processed_frames

def extract_audio_features(video_path):

    directory, file = os.path.split(video_path)
    folder_name = os.path.basename(directory)
    file_name = str(file).replace('.mp4','')
    training_root_dir = 'D:\Codebase\lip_reading_project_public\sentence_lip_reading\lrs2_training_video_dir'
    # Use ffmpeg to extract frames from the video
    print("Extracting frames from the video: " + video_path)
    audio_filename = file_name + '.wav'
    audio_dir = os.path.join(training_root_dir,folder_name,file_name,'audio')
    os.makedirs(audio_dir, exist_ok=True)

    video = VideoFileClip(video_path)
    audio = video.audio
    audio_file_path = os.path.join(audio_dir,audio_filename)
    audio.write_audiofile(audio_file_path)
    audio, sr = librosa.load(audio_file_path, sr=16000)  # Adjust sample rate as needed
    # Extract audio features (e.g., MFCCs, spectrograms, etc.)
    return audio

# Initialize the model
num_classes = 2  # Example, replace with your actual number of classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

visual_test_data = [
    [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])],  # Frames of video 1
    [np.array([[13, 14, 15], [16, 17, 18]]), np.array([[19, 20, 21], [22, 23, 24]])],  # Frames of video 2
    # Frames of other videos...
]


# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    # Load training data
    data_dir = 'train_data'  # Directory containing training data
    video_files, transcriptions = load_training_data('D:\Codebase\lip_reading_project_public\sentence_lip_reading\lrwinputlist_1.txt')
    print("Number for videos for training: " + str(len(video_files)))
    print("Number for transcriptions for training: " + str(len(transcriptions)))
    visual_data = []
    audio_data = []
    for video_file in video_files:
        frames = preprocess_video(video_file)
        audio_features = extract_audio_features(video_file)
        visual_data.append(frames)
        audio_data.append(audio_features)

    # max_length = max(len(seq) for seq in visual_data)
    # Pad or truncate sequences to the maximum length
    # padded_visual_data = [seq + [0] * (max_length - len(seq)) for seq in visual_data]
    # visual_tensor = torch.tensor(padded_visual_data)

    max_length = max(len(seq) for seq in audio_data)
    padded_audio_data = [np.pad(seq, (0, max_length - len(seq))) for seq in audio_data]
    audio_tensor = torch.tensor(padded_audio_data)
    # audio_data = torch.tensor(audio_data)
    max_frames = max(len(video) for video in visual_data)

    padded_visual_data = [
        video + [np.zeros_like(video[0])] * (max_frames - len(video))
        if len(video) < max_frames
        else video[:max_frames]
        for video in visual_data
    ]
    # Convert padded_visual_data to a tensor
    visual_tensor = torch.tensor(padded_visual_data)

    max_length = 10  # Update with the desired maximum length

    # Tokenize each transcription and pad or truncate it to the maximum length
    encoded_transcriptions = [tokenizer.encode(transcription, add_special_tokens=True, max_length=max_length, truncation=True) for transcription in transcriptions]

    # Pad or truncate the token sequences to ensure consistency
    padded_encoded_transcriptions = [seq + [0] * (max_length - len(seq)) for seq in encoded_transcriptions]

    # Convert the list of padded tokenized transcriptions into a tensor
    labels_tensor = torch.tensor(padded_encoded_transcriptions)

    train_dataset = TensorDataset(visual_tensor, audio_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch in train_loader:
        optimizer.zero_grad()
        visual_inputs, audio_inputs, labels = batch
        # Preprocess visual and audio inputs (e.g., tokenize text, normalize audio features)
        # Concatenate visual and audio inputs
        # Step 1: Check the shapes
        print("Visual inputs shape:", visual_inputs.shape)
        print("Audio inputs shape:", audio_inputs.shape)

        audio_inputs = audio_inputs.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        combined_inputs = torch.cat((visual_inputs, audio_inputs), dim=1)

        inputs = tokenizer(combined_inputs, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# Save the trained model
model_file = 'lrwfirst_model.pth'
torch.save(model.state_dict(), model_file)
print(f'Model %s is created successful ',model_file)
print("Terminating...")
exit(0)
# Step 6: Prediction

# Load the saved model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model.load_state_dict(torch.load('lip_reading_model.pth'))
model.eval()

# Predict text from video
def predict_text(video_path):
    frames = preprocess_video(video_path)
    audio_features = extract_audio_features(video_path)
    inputs = tokenizer(frames, return_tensors='pt', padding=True, truncation=True)
    audio_inputs = tokenizer(audio_features, return_tensors='pt', padding=True, truncation=True)
    combined_inputs = {key: torch.cat((inputs[key], audio_inputs[key]), dim=1) for key in inputs}
    with torch.no_grad():
        outputs = model(**combined_inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        predicted_text = tokenizer.decode(predictions)
        return predicted_text

exit(0)
# Example usage
video_path = 'example_video.mp4'
predicted_text = predict_text(video_path)
print(predicted_text)
