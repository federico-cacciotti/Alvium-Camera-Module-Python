import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os
from tqdm import tqdm

data_path = os.path.join(os.path.dirname(__file__), 'data')
default_data_path = os.path.join(data_path, 'current')
parser = argparse.ArgumentParser(description='Decode and display a raw frame from the Alvium camera.')
parser.add_argument('-i', '--input', type=str, default=default_data_path, help='Path to the raw frame file or folder containing raw frames')

def main():
    args = parser.parse_args()
    path_to_raw = args.input
    print(f'Input path: {path_to_raw}')

    # if the input is a folder, get all raw files in it
    if os.path.isdir(path_to_raw):
        default_output_path = os.path.join(path_to_raw, 'decoded')
        os.makedirs(default_output_path, exist_ok=True)
        print(f'Output path: {default_output_path}')

        raw_files = glob(os.path.join(path_to_raw, '*.raw'))
        if not raw_files:
            print("No raw files found in the specified folder.")
            return
        for raw_file in tqdm(raw_files):
            with open(raw_file, 'rb') as f:
                raw = f.read()
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((3008, 4128))
            
            debayered = cv2.cvtColor(arr, cv2.COLOR_BAYER_RG2BGR)

            # save tiff frame
            output_path = os.path.join(default_output_path, os.path.basename(raw_file).rsplit('.', 1)[0] + '.tiff')
            cv2.imwrite(output_path, debayered)
        print(f'Saved debayered frames to {output_path}')
    
    else:
        # take the parent directory of the raw file as the default output path
        default_output_path = os.path.dirname(path_to_raw)
        default_output_path = os.path.join(default_output_path, 'decoded')
        os.makedirs(default_output_path, exist_ok=True)
        print(f'Output path: {default_output_path}')

        with open(path_to_raw, 'rb') as f:
            raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((3008, 4128))
        
        debayered = cv2.cvtColor(arr, cv2.COLOR_BAYER_RG2BGR)

        # save tiff frame
        output_path = os.path.join(default_output_path, os.path.basename(path_to_raw).rsplit('.', 1)[0] + '.tiff')
        cv2.imwrite(output_path, debayered)
        print(f'Saved debayered frame to {output_path}')

if __name__ == '__main__':
    main()