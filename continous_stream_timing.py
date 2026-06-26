import queue
import threading
from Camera import Camera
import os
import matplotlib.pyplot as plt
import time
import logging

CAMERA_ID = "DEV_1AB22C0707F3"
ACQUISITION_DURATION_S = 120  # seconds
EXPOSURE_TIME_US = 10000  # microseconds
GAIN = 30  # dB
FRAMERATE = 15  # frames per second

data_path = os.path.join(os.path.dirname(__file__), 'data')
output_dir = time.strftime(f'%Y%m%d_%H%M%S_T{ACQUISITION_DURATION_S}_E{EXPOSURE_TIME_US}_G{GAIN}_F{FRAMERATE}')
output_dir = os.path.join(data_path, output_dir)
# make data directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)
# make output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# define a queue to hold frames for writing to disk
frame_queue = queue.Queue()

# create a symbolic link to the "current" folder, if it exits, remove it first
link_path = os.path.join(data_path, 'current')
if os.path.islink(link_path):
    os.unlink(link_path)
os.symlink(os.path.abspath(output_dir), link_path)


def main():
    with Camera(CAMERA_ID, output_dir=output_dir) as camera:
        values_dict = {
            'ExposureAuto': 'Off',
            'GainAuto': 'Off',
            'BalanceWhiteAuto': 'Off',
            'Gamma': 1.0,
            'Hue': 0.0,
            'Saturation': 1.0,
            'Sharpness': 0.0,
            'SensorShutterMode': 'GlobalShutter',
            'ExposureTime': EXPOSURE_TIME_US,
            'Gain': GAIN,
            'OffsetX': 0,
            'OffsetY': 0,
            'Width': camera.get_value('WidthMax'),
            'Height': camera.get_value('HeightMax'),
            'BinningHorizontal': 1,
            'BinningVertical': 1,
            'ReverseX': False,
            'ReverseY': False,
            'PixelFormat': 'BayerRG8',
            'SensorBitDepth': 'Bpp12',
            'AdaptiveNoiseSuppressionFactor': 1.0,
            'AcquisitionFrameRateEnable': True,
            'AcquisitionFrameRate': FRAMERATE
        }

        def writer(q):
            i = 0
            while True:
                frame = q.get()
                if frame is None:
                    break
                with camera.timed_write():          # records disk write duration
                    arr = frame.as_numpy_ndarray().tobytes()
                    path = f'{output_dir}/frame_{i:06d}.raw'
                    with open(path, 'wb') as f:
                        f.write(arr)
                    camera.logger.info(f'Wrote frame {i}')
                i += 1

        camera.set_values(values_dict)
        camera.log_all_features()

        t = threading.Thread(target=writer, args=(frame_queue,), daemon=True)
        t.start()

        camera.start_continuous_streaming(
            callback=lambda f: frame_queue.put(f),
            timing=True,
            monitored_queue=frame_queue   # queue depth sampled at every frame
        )
        
        time.sleep(ACQUISITION_DURATION_S)

        camera.stop_continuous_streaming()

        frame_queue.put(None)
        t.join()

        stats = camera.get_streaming_stats()
        print(f"Frames:          {stats['n_frames']}")
        print(f"Acquisition fps: {stats['mean_fps']:.2f} ± {stats['std_fps']:.2f}")
        print(f"Total duration:  {stats['total_duration_s']:.3f} s")
        print(f"Write duration:  {stats['mean_write_s']*1e3:.1f} ± {stats['std_write_s']*1e3:.1f} ms  (max {stats['max_write_s']*1e3:.1f} ms)")
        print(f"Write throughput:{stats['mean_write_fps']:.2f} fps")
        print(f"Queue overflow:  {stats['queue_overflow_ratio']:.1%} of frames")
        print(f"Max queue depth: {stats['max_queue_size']}")

        if stats is not None:
            fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
            axs[0][0].plot(1 / stats['intervals_s'], color='gray', label='FPS')
            axs[0][0].axhline(stats['mean_fps'], color='black', linestyle='--', label='Mean FPS')
            axs[0][0].set_xlabel('Frame index')
            axs[0][0].set_ylabel('FPS')
            axs[0][0].legend()

            axs[0][1].plot((stats['write_timestamps_s']-stats['write_timestamps_s'][0]), stats['write_durations_s'], color='gray')
            axs[0][1].axhline(stats['mean_write_s'], color='black', linestyle='--', label='Mean write duration')
            axs[0][1].axhline(stats['median_write_s'], color='red', linestyle='--', label='Median write duration')
            axs[0][1].axvspan(stats['host_timestamps_s'][0]-stats['write_timestamps_s'][0], stats['host_timestamps_s'][-1]-stats['write_timestamps_s'][0], color='blue', alpha=0.1, label='Stream time')
            axs[0][1].set_xlabel('Time (s)')
            axs[0][1].set_ylabel('Frame write duration (s)')
            axs[0][1].legend()

            axs[1][1].plot(stats['queue_sizes'], color='gray')
            axs[1][1].axhline(stats['mean_queue_size'], color='black', linestyle='--', label='Mean queue size')
            axs[1][1].axhline(stats['median_queue_size'], color='red', linestyle='--', label='Median queue size')
            axs[1][1].set_xlabel('Frame index')
            axs[1][1].set_ylabel('Queue size (frames)')
            axs[1][1].legend()

            axs[1][0].plot((stats['write_timestamps_s']-stats['write_timestamps_s'][0]), 1 / stats['write_durations_s'], color='gray')
            axs[1][0].axhline(1 / stats['mean_write_s'], color='black', linestyle='--', label='Mean write FPS')
            axs[1][0].axhline(1 / stats['median_write_s'], color='red', linestyle='--', label='Median write FPS')
            axs[1][0].axvspan(stats['host_timestamps_s'][0]-stats['write_timestamps_s'][0], stats['host_timestamps_s'][-1]-stats['write_timestamps_s'][0], color='blue', alpha=0.1, label='Stream time')
            axs[1][0].set_xlabel('Time (s)')
            axs[1][0].set_ylabel('Write throughput (fps)')
            axs[1][0].legend()

            axs[0][2].plot(stats['cpu_usage_percent'], color='gray')
            axs[0][2].set_xlabel('Frame index')
            axs[0][2].set_ylabel('CPU usage (%)')

            axs[1][2].plot(stats['ram_usage_percent'], color='gray')
            axs[1][2].set_xlabel('Frame index')
            axs[1][2].set_ylabel('RAM usage (%)')

            fig.savefig(os.path.join(output_dir, 'streaming_stats.png'))
            plt.show()


if __name__ == '__main__':
    main()