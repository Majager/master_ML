import os

# Split the data set into training and second set based on indices
def split_data(data,labels,recording_ids,first_indices,second_indices):
    first_data = [data[i] for i in first_indices]
    first_labels = [labels[i] for i in first_indices]
    first_recording_ids = [recording_ids[i] for i in first_indices]

    second_data = [data[i] for i in second_indices]
    second_labels = [labels[i] for i in second_indices]
    second_recording_ids = [recording_ids[i] for i in second_indices]
    return first_data, first_labels, first_recording_ids, second_data, second_labels, second_recording_ids

def store_results_filename(test_name,timestamp):
    results_path = f"Results\\{test_name}"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    data_path = os.path.join(results_path,"data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    r_path = os.path.join(data_path,timestamp)
    if not os.path.exists(r_path):
        os.mkdir(r_path)
    return r_path