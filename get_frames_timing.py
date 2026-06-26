import cv2
from Camera import Camera
import os
import matplotlib.pyplot as plt

CAMERA_ID = "DEV_1AB22C0707F3"
test_frames_dir = 'test_frames'
os.makedirs(test_frames_dir, exist_ok=True)

def main():
    with Camera(CAMERA_ID) as camera:

        # set initial camera parameters
        values_dict = {
            'ExposureAuto': 'Off',
            'GainAuto': 'Off',
            'BalanceWhiteAuto': 'Off',
            'Gamma': 1.0,
            'Hue': 0.0,
            'Saturation': 1.0,
            'Sharpness': 0.0,
            'SensorShutterMode': 'GlobalShutter',
            'ExposureTime': 1000,
            'Gain': 0,
            'OffsetX': 0,
            'OffsetY': 0,
            'Width': camera.get_value('WidthMax'),
            'Height': camera.get_value('HeightMax'),
            'BinningHorizontal': 1,
            'BinningVertical': 1,
            'ReverseX': False,
            'ReverseY': False,
            'PixelFormat': 'BayerRG12',
            'SensorBitDepth': 'Bpp12',
            'AdaptiveNoiseSuppressionFactor': 1.0,
            'AcquisitionFrameRateEnable': False,
            'AcquisitionFrameRate': 5
        }
        camera.set_values(values_dict)

        # save txt file with all camera features and their values
        camera.log_all_features()

        # acquire a single frame and save it as a raw file
        frame = camera.get_frame()
        if frame is not None:
            raw = frame.as_numpy_ndarray()
            cv2.imwrite(f'{test_frames_dir}/single_frame.tiff', raw)

        # acquire several frames and save them as raw files
        frames, get_frames_timing_info = camera.get_frames(100, fps=None, timing=True)
        for i, frame in enumerate(frames):
            raw = frame.as_numpy_ndarray()
            cv2.imwrite(f'{test_frames_dir}/frame_{i:03d}.tiff', raw)

    # plot timing information
    if get_frames_timing_info is not None:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(1 / get_frames_timing_info['intervals_s'], label='FPS', color='gray')
        ax.axhline(get_frames_timing_info['mean_fps'], label='Mean FPS', color='black', linestyle='--')
        ax.set_xlabel('Frame number')
        ax.set_ylabel('FPS')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    main()