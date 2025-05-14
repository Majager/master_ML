import librosa
import numpy as np
from matplotlib.colors import Normalize
import pickle
import json
import scipy.signal
from sklearn.preprocessing import StandardScaler

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

# Extract features
# zero_crossings: measures noisiness
# rms_energy: measures loudness over time
# mfcc:  extract frequency bands over time
# spectral_centroid: determines brightness
# spectral_bandwidth: determines spread of frequencies
# spectral contrast: 
# spectral rolloff: 
# spectral flatness:
def convert_to_features(data_segments, sr, n_fft, hop_length,n_mfcc):
    features = []
    for i in range(len(data_segments)):
        features_segment = []
        # Time domain features
        zero_crossings_rate = np.mean(librosa.feature.zero_crossing_rate(data_segments[i]))
        rms_energy = np.mean(librosa.feature.rms(y=data_segments[i]))

        # Frequency domain features
        mel_spectrogram = librosa.feature.melspectrogram(y = data_segments[i], sr = sr, n_fft = n_fft, hop_length = hop_length)
        db_spectrogram = librosa.power_to_db(mel_spectrogram, ref = np.max)
        mfcc = librosa.feature.mfcc(S=db_spectrogram,sr=sr,n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data_segments[i],sr=sr,hop_length=hop_length,n_fft=n_fft))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data_segments[i],sr=sr,hop_length=hop_length,n_fft=n_fft)) 
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data_segments[i],sr=sr,hop_length=hop_length,n_fft=n_fft))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data_segments[i],sr=sr,hop_length=hop_length,n_fft=n_fft))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=data_segments[i],hop_length=hop_length,n_fft=n_fft))
        
        features_segment = [zero_crossings_rate, rms_energy, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, spectral_flatness]
        features_segment.extend(np.mean(mfcc, axis=1))
        features_segment.extend(np.mean(mfcc_delta, axis=1))
        features_segment.extend(np.mean(mfcc_delta2, axis=1))

        features.append(features_segment)
    
    features = np.array(features, dtype=np.float32)

    normalize = StandardScaler()
    normalized_features = (normalize.fit_transform(features))
    return normalized_features

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

def notch_filter(data,sr,notch_freq=50,quality_factor=30):
    b, a = scipy.signal.iirnotch(notch_freq,quality_factor,sr)
    return scipy.signal.filtfilt(b,a,data)

# Feature extraction 
def feature_extraction(file_paths, meta_data_files, file_ids, position, segment_length, overlap_length, sr, n_fft, hop_length, n_mfcc, update_features = True, multiclass = False):
    features = []
    labels = []
    recording_ids = []

    if (update_features):
        num_subjects = len(file_paths)
        features = np.empty(num_subjects,dtype=object)
        labels = np.empty(num_subjects,dtype=object)
        for i, file_path in enumerate(file_paths,0):
            # Sampling of the wav file, resampled to sr in function

            # Preprocessing
            (data, sr) = librosa.load(file_path, sr = sr, mono = True, dtype = np.float32)
            data = notch_filter(data,sr)
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

            # Extract features for ML model
            recording_features = convert_to_features(data_segments,sr,n_fft, hop_length,n_mfcc)
            features[i] = recording_features

        recording_ids = file_ids

        # Store features for later use
        with open(f'features_{position}_{segment_length}.pickle', 'wb') as handle:
            pickle.dump([features,labels, recording_ids],handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if multiclass:
            # Extract features from previous calculations
            with open(f'features_{position}_{segment_length}_multiclass.pickle', 'rb') as handle:
                features,labels, recording_ids = pickle.load(handle)
        else: 
            # Extract features from previous calculations
            with open(f'features_{position}_{segment_length}.pickle', 'rb') as handle:
                features,labels, recording_ids = pickle.load(handle)
    
    return features,labels, recording_ids

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV

