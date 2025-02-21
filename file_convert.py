import os
import glob

def find_data(root_folder_path,position):
    wav_files = []
    metadata_files = []
    recording_ids = []
    # Loop through each year folder
    for year_folder in os.listdir(root_folder_path):
        year_folder_path = os.path.join(root_folder_path,year_folder)
        # Loop through each participant in data folder
        for subject_folder in os.listdir(year_folder_path):
            subject_folder_path = os.path.join(year_folder_path,subject_folder)
            # Ensure directory (folder)
            if os.path.isdir(subject_folder_path): 
                # Loop through each recording in subject
                for recording_folder in os.listdir(subject_folder_path):
                    recording_folder_path = os.path.join(subject_folder_path,recording_folder)
                    if os.path.isdir(recording_folder_path):
                        metadata_file_path = os.path.join(recording_folder_path, f"{recording_folder}.json")
                        if (os.path.exists(metadata_file_path)):
                            # Search for AUDIO folder for subject
                            audio_folder_path = os.path.join(recording_folder_path,'AUDIO')
                            if os.path.isdir(audio_folder_path):
                                # Look for .wav file from specific position
                                vector = position.split("-")
                                for pos in vector:
                                    wav_file_path = os.path.join(audio_folder_path,f'{pos}.wav')
                                    if os.path.exists(wav_file_path):
                                        wav_files.append(wav_file_path)
                                        metadata_files.append(metadata_file_path)
                                        recording_ids.append(recording_folder)
    return wav_files, metadata_files, recording_ids