from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import librosa


def load_training_data(driverfile):
    video_files = []
    transcriptions = []
    with open(driverfile, 'r') as f:
        for line in f:
            video_file, transcription = line.strip().split('|')
            video_files.append(video_file)
            transcriptions.append(transcription)
    return video_files, transcriptions

def get_video_frames(framesDirectory):
    return ""

def get_audio(audioFilePath):
    audio, sr = librosa.load(audioFilePath, sr=16000)
    return audio

def preprocess_text(text_content):
    # Use LabelEncoder to convert text labels to numerical labels
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(text_content)

    # Use to_categorical to convert numerical labels to one-hot encoded vectors
    one_hot_encoded = to_categorical(integer_encoded)

    return one_hot_encoded, label_encoder.classes