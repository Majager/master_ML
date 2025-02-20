import librosa
import numpy as np
from matplotlib.colors import Normalize
import pickle
import json
import scipy.signal

# https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
# https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html

# Code to extract number from filename
# def extract_number(filename):
#     # Function to extract the number from the filename
#     basename = os.path.basename(filename)
#     number = ''.join(filter(str.isdigit, basename))
#     return int(number) if number else 0
# wav_files = sorted(glob.glob(os.path.join(folder_path, '*.wav')),key=file_convert_labels.extract_number)

# Code for splitting audio data in different segments
def split_data_in_segments(data, segment_length, overlap_length, sr):
    segment_samples = segment_length * sr
    overlap_samples = overlap_length * sr
    step_samples = segment_samples - overlap_samples

    segments = np.empty((int((len(data)-segment_samples)//step_samples + 1),int(segment_samples)),dtype=np.float32)
    
    for i in range(len(segments)):
        segments[i] = data[int(i*step_samples):int(i*step_samples+segment_samples)]

    return segments

# Code for converting data to spectrograms
def convert_to_spectrograms(data_segments, sr, n_fft, hop_length):
    spectrograms = []
    for i in range(len(data_segments)):
        mel_spectrogram = librosa.feature.melspectrogram(y = data_segments[i], sr = sr, n_fft = n_fft, hop_length = hop_length)
        db_spectrogram = librosa.power_to_db(mel_spectrogram, ref = np.max)
        norm = Normalize(vmin=np.min(db_spectrogram), vmax=np.max(db_spectrogram))
        norm_spectrogram = norm(db_spectrogram)
        spectrograms.append(norm_spectrogram)
    spectrograms = np.array(spectrograms, dtype=np.float32)
    return spectrograms

# Code for converting data to MFCC
def convert_to_MFCCs(data_segments, sr, n_fft, hop_length,n_mfcc):
    MFCCs = []
    for i in range(len(data_segments)):
        mel_spectrogram = librosa.feature.melspectrogram(y = data_segments[i], sr = sr, n_fft = n_fft, hop_length = hop_length)
        db_spectrogram = librosa.power_to_db(mel_spectrogram, ref = np.max)
        MFCC = librosa.feature.mfcc(S=db_spectrogram,sr=sr,n_mfcc=n_mfcc)
        MFCCs.append(np.mean(MFCC, axis=1))
    MFCCs = np.array(MFCCs, dtype=np.float32)
    return MFCCs

# Find labels for meal based on json file
def convert_labels(meal_start, meal_end, segment_length, overlap_length, sr):
    segment_samples = segment_length * sr
    overlap_samples = overlap_length * sr
    step_samples = segment_samples - overlap_samples
    meal_start_samples = meal_start*sr
    meal_end_samples = meal_end*sr

    meal_start_segment = (meal_start_samples-overlap_samples)//step_samples
    meal_end_segment = (meal_end_samples-overlap_samples)//step_samples

    return meal_start_segment, meal_end_segment

# Feature extraction 
def feature_extraction(file_paths, meta_data_files, position, segment_length, overlap_length, sr, n_fft, hop_length, n_mfcc, update_features = True):
    num_subjects = len(file_paths)
    features = np.empty(num_subjects,dtype=object)
    labels = np.empty(num_subjects,dtype=object)

    if (update_features):
        for i, file_path in enumerate(file_paths,0):
            # Sampling of the wav file, resampled to sr in function
            (data, sr) = librosa.load(file_path, sr = sr, mono = True, dtype = np.float32)
            data_segments = split_data_in_segments(data, segment_length, overlap_length, sr)

            # Extract labels from metadata files
            meta_data = ""
            audio_offset = 0
            with open(meta_data_files[i], 'r') as file:
                meta_data = json.load(file)
            if "audio_offset" in meta_data:
                audio_offset = meta_data["audio_offset"]
            meal_start_segment, meal_end_segment = convert_labels(meta_data["mealStart"]-audio_offset,meta_data["mealEnd"]-audio_offset,segment_length,overlap_length,sr)
            data_labels = np.zeros(len(data_segments),dtype=int)
            data_labels[int(meal_start_segment):int(meal_end_segment)] = 1
            labels[i] = data_labels

            # Extract MFCC as features for ML model
            data_mfcc = convert_to_MFCCs(data_segments,sr,n_fft, hop_length,n_mfcc)
            features[i] = data_mfcc

        # Store features for later use
        with open(f'features_mfcc_{position}.pickle', 'wb') as handle:
            pickle.dump([features,labels],handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Extract features from previous calculations
        with open(f'features_mfcc_{position}.pickle', 'rb') as handle:
            features,labels = pickle.load(handle)
    
    return features,labels


