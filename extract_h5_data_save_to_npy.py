import os
import numpy as np
import zipfile
import tempfile
from KKT_Module.KKTUtility.H5Tool import KKTH5Tool 

def extract_and_save_data_labels_from_h5(filename, data_save_path, labels_save_path):
    h5_tool = KKTH5Tool()
    
    data, label, axis = h5_tool.readData(filename)
    
    print(f"Data shape: {data.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Axis shape: {axis.shape}")
    
    np.save(data_save_path, data)
    np.save(labels_save_path, label)
    
    print(f"Data saved to {data_save_path}")
    print(f"Labels saved to {labels_save_path}")
    
    return data, label, axis

def process_h5_file(h5_file_path):
    data_save_path = f"./data_{os.path.basename(h5_file_path)}.npy"
    labels_save_path = f"./labels_{os.path.basename(h5_file_path)}.npy"

    try:
        data, label, axis = extract_and_save_data_labels_from_h5(h5_file_path, data_save_path, labels_save_path)

        print(f"Data shape: {data.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Axis shape: {axis.shape}")

        return data_save_path, labels_save_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def process_zip_file(zip_path, output_dir):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)

            npy_files = []

            for root, dirs, files in os.walk(tmpdirname):
                for file in files:
                    if file.endswith('.h5'):
                        category = os.path.relpath(root, tmpdirname)
                        h5_file_path = os.path.join(root, file)
                        data_save_path = os.path.join(tmpdirname, f"{category}/data_{file}.npy")
                        labels_save_path = os.path.join(tmpdirname, f"{category}/labels_{file}.npy")

                        os.makedirs(os.path.dirname(data_save_path), exist_ok=True)

                        extract_and_save_data_labels_from_h5(h5_file_path, data_save_path, labels_save_path)
                        npy_files.append(data_save_path)
                        npy_files.append(labels_save_path)

            output_zip_path = os.path.join(output_dir, "extracted_data_and_labels.zip")
            with zipfile.ZipFile(output_zip_path, 'w') as zipf:
                for npy_file in npy_files:
                    arcname = os.path.relpath(npy_file, tmpdirname)
                    zipf.write(npy_file, arcname)

            print(f"Processed ZIP file saved to {output_zip_path}")

            return output_zip_path
    except Exception as e:
        print(f"An error occurred while processing the ZIP file: {e}")
        return None

zip_file_path = "RDIPHD.zip"
output_directory = "./output" 
os.makedirs(output_directory, exist_ok=True)
output_zip_path = process_zip_file(zip_file_path, output_directory)
