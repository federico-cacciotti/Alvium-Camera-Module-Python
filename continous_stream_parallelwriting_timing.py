import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from Camera import Camera
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

CAMERA_ID  = "DEV_1AB22C0707F3"
N_WORKERS  = 2           # number of parallel writer threads; tune for your NVMe
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

def _write_frame(arr, path, camera):
    """Write a single frame to disk, recording duration via camera.timed_write()."""
    with camera.timed_write():
        with open(path, 'wb') as f:
            f.write(arr)


def dispatcher_thread(frame_queue, output_dir, camera, n_workers):
    """Pull frames from the queue and dispatch writes to the thread pool.

    Each frame's numpy array is extracted here (in the dispatcher thread)
    before submission so the SDK frame buffer is not held by a worker.
    Workers only touch the already-copied numpy array.
    """
    futures = []
    i = 0
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix='nv-writer') as executor:
        while True:
            frame = frame_queue.get()
            if frame is None:   # poison pill
                break
            arr  = frame.as_numpy_ndarray().tobytes()
            path = f'{output_dir}/frame_{i:06d}.raw'
            futures.append(executor.submit(_write_frame, arr, path, camera))
            i += 1
        # executor.__exit__ waits for all submitted futures before returning

    # propagate any worker exceptions
    for fut in futures:
        exc = fut.exception()
        if exc:
            raise exc


def main():
    
    with Camera(CAMERA_ID) as camera:
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

        camera.set_values(values_dict)
        camera.log_all_features()

        dispatcher = threading.Thread(
            target=dispatcher_thread,
            args=(frame_queue, output_dir, camera, N_WORKERS),
            daemon=True,
            name='dispatcher',
        )
        dispatcher.start()

        camera.start_continuous_streaming(
            callback=lambda f: frame_queue.put(f),
            timing=True,
            monitored_queue=frame_queue,
        )

        time.sleep(ACQUISITION_DURATION_S)

        camera.stop_continuous_streaming()

        frame_queue.put(None)   # poison pill: stop the dispatcher
        dispatcher.join()       # wait for all in-flight writes to finish

        stats = camera.get_streaming_stats()

    # print summary 
    print(f"Workers          : {N_WORKERS}")
    print(f"Frames received  : {stats['n_frames']}")
    print(f"Acquisition fps  : {stats['mean_fps']:.2f} ± {stats['std_fps']:.2f}  (median {stats['median_fps']:.2f})")
    print(f"Total duration   : {stats['total_duration_s']:.3f} s")
    print(f"Writes           : {stats['n_writes']}")
    print(f"Write latency    : {stats['mean_write_s']*1e3:.1f} ± {stats['std_write_s']*1e3:.1f} ms  (max {stats['max_write_s']*1e3:.1f} ms, median {stats['median_write_s']*1e3:.1f} ms)")
    print(f"Write throughput : {stats['mean_write_fps']:.2f} fps")
    print(f"Queue overflow   : {stats['queue_overflow_ratio']:.1%} of frames  (max depth {stats['max_queue_size']})")

    # plots
    wt = stats['write_timestamps_s']
    ht = stats['host_timestamps_s']

    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(f'Parallel writing — {N_WORKERS} workers', fontsize=13)

    # acquisition fps over time
    axs[0][0].plot(1.0 / stats['intervals_s'], color='gray', label='FPS')
    axs[0][0].axhline(stats['mean_fps'],   color='black', linestyle='--', label='Mean FPS')
    axs[0][0].axhline(stats['median_fps'], color='red',   linestyle='--', label='Median FPS')
    axs[0][0].set_xlabel('Frame index')
    axs[0][0].set_ylabel('Acquisition FPS')
    axs[0][0].legend()

    # write duration over wall-clock time
    axs[0][1].plot(wt - wt[0], stats['write_durations_s'], color='gray')
    axs[0][1].axhline(stats['mean_write_s'],   color='black', linestyle='--', label='Mean')
    axs[0][1].axhline(stats['median_write_s'], color='red',   linestyle='--', label='Median')
    axs[0][1].axvspan(ht[0] - wt[0], ht[-1] - wt[0],
                      color='blue', alpha=0.1, label='Stream time')
    axs[0][1].set_xlabel('Time (s)')
    axs[0][1].set_ylabel('Write duration (s)')
    axs[0][1].legend()

    # queue size over frame index
    axs[1][1].plot(stats['queue_sizes'], color='gray')
    axs[1][1].axhline(stats['mean_queue_size'],   color='black', linestyle='--', label='Mean')
    axs[1][1].axhline(stats['median_queue_size'], color='red',   linestyle='--', label='Median')
    axs[1][1].set_xlabel('Frame index')
    axs[1][1].set_ylabel('Queue depth (frames)')
    axs[1][1].legend()

    # instantaneous write throughput (fps) over wall-clock time
    axs[1][0].plot(wt - wt[0], 1.0 / stats['write_durations_s'], color='gray')
    axs[1][0].axhline(1.0 / stats['mean_write_s'],   color='black', linestyle='--', label='Mean write FPS')
    axs[1][0].axhline(1.0 / stats['median_write_s'], color='red',   linestyle='--', label='Median write FPS')
    axs[1][0].axvspan(ht[0] - wt[0], ht[-1] - wt[0],
                      color='blue', alpha=0.1, label='Stream time')
    axs[1][0].set_xlabel('Time (s)')
    axs[1][0].set_ylabel('Write throughput (fps)')
    axs[1][0].legend()

    # CPU usage
    axs[0][2].plot(stats['cpu_usage_percent'], color='gray')
    axs[0][2].set_xlabel('Frame index')
    axs[0][2].set_ylabel('CPU usage (%)')
    axs[0][2].set_title(f'CPU  (mean {np.mean(stats["cpu_usage_percent"]):.1f}%)')

    # RAM usage
    axs[1][2].plot(stats['ram_usage_percent'], color='gray')
    axs[1][2].set_xlabel('Frame index')
    axs[1][2].set_ylabel('RAM usage (%)')
    axs[1][2].set_title(f'RAM  (mean {np.mean(stats["ram_usage_percent"]):.1f}%)')

    fig.savefig('streaming_stats_parallel.png')
    plt.show()


if __name__ == '__main__':
    main()
