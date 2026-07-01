import queue
import sys
import numpy as np
import time
import threading
import logging
from contextlib import contextmanager
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import vmbpy
import psutil
import os

from . import parameters as params


class Camera:
    """
    Wraps a VmbPy camera as a context manager.

    VmbSystem must remain open for the entire lifetime of any camera handle.
    Opening and closing separate VmbSystem contexts while holding a camera
    object causes segfaults because the underlying C handles become invalid.
    A single VmbSystem context is therefore opened in __enter__ and kept alive
    until __exit__.

    Usage:
        with Camera(camera_id) as camera:
            camera.set_values({...})
            frame = camera.get_frame()
    """

    def __init__(self, camera_id=params.CAMERA_ID, output_path=params.DEFAULT_OUTPUT_PATH, writing_threads=params.DEFAULT_WRITING_THREADS, settings={}):

        self.camera_id = camera_id
        self.output_path = output_path

        # create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # configure logging
        # VmbPy logs every internal error via a logger named 'vmbpyLog'. With propagate=True
        # (Python default) those records bubble up to the root logger and pollute the output.
        # We only want our own application-level logs, so disable propagation for vmbpy's logger.
        logging.getLogger('vmbpyLog').propagate = False

        # check if root logger already exists, if not, configure it with defaults from parameters.py
        if not logging.getLogger().hasHandlers():
            logger = logging.getLogger()
            logger.setLevel(getattr(logging, params.LOGGING_LEVEL))
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(params.LOGGING_FORMAT, params.LOGGING_DATEFMT))
            logger.addHandler(handler)

        self.logger = logging.getLogger(__name__)

        # define second logger for frame timings
        timestamp_logger_filename = os.path.join(output_path, params.LOGGING_FILENAME) if output_path else params.LOGGING_FILENAME
        self.timestamp_logger = logging.getLogger('frame_timestamps')
        self.timestamp_logger.setLevel(getattr(logging, params.LOGGING_LEVEL))
        timestamp_handler = logging.FileHandler(timestamp_logger_filename)
        timestamp_handler.setFormatter(logging.Formatter(params.LOGGING_FORMAT, params.LOGGING_DATEFMT))
        self.timestamp_logger.addHandler(timestamp_handler)

        self.writing_threads = writing_threads
        self._vmb_instance = vmbpy.VmbSystem.get_instance()
        self._vmb = None   # set in __enter__
        self.cam = None    # set in __enter__
        self.features = None
        # define a queue to hold frames for writing to disk
        self.frame_queue = queue.Queue()
        self.dispatcher = None

        self.settings = settings

    def __enter__(self):
        self._vmb = self._vmb_instance.__enter__()
        self.cam = self._open_camera()
        self.cam.__enter__()
        self.logger.info(f'Camera opened: {self.cam.get_id()} - {self.cam.get_model()}')
        self.logger.info(f'Output path: {self.output_path}')
        self.logger.info(f'Writing threads: {self.writing_threads}')
        self.features = self._read_all_features()

        self.logger.info('Setting camera features...')
        values_dict = {
            'ExposureAuto': 'Off',
            'GainAuto': 'Off',
            'BalanceWhiteAuto': 'Off',
            'Gamma': 1.0,
            'Hue': 0.0,
            'Saturation': self.settings.get('saturation', params.DEFAULT_SATURATION) if self.settings.get('saturation', params.DEFAULT_SATURATION) is not None else params.DEFAULT_SATURATION,
            'Sharpness': self.settings.get('sharpness', params.DEFAULT_SHARPNESS_FACTOR) if self.settings.get('sharpness', params.DEFAULT_SHARPNESS_FACTOR) is not None else params.DEFAULT_SHARPNESS_FACTOR,
            'SensorShutterMode': 'GlobalShutter',
            'ExposureTime': self.settings.get('exposure', params.DEFAULT_EXPOSURE_TIME_US) if self.settings.get('exposure', params.DEFAULT_EXPOSURE_TIME_US) is not None else params.DEFAULT_EXPOSURE_TIME_US,
            'Gain': self.settings.get('gain', params.DEFAULT_GAIN) if self.settings.get('gain', params.DEFAULT_GAIN) is not None else params.DEFAULT_GAIN,
            'OffsetX': params.DEFAULT_ROI_X_OFFSET,
            'OffsetY': params.DEFAULT_ROI_Y_OFFSET,
            'Width':   self.get_value('WidthMax'),
            'Height':  self.get_value('HeightMax'),
            'BinningHorizontal': params.DEFAULT_BINNING_SIZE_X,
            'BinningVertical': params.DEFAULT_BINNING_SIZE_Y,
            'ReverseX': params.DEFAULT_REVERSE_X,
            'ReverseY': params.DEFAULT_REVERSE_Y,
            'PixelFormat': self.settings.get('format', params.DEFAULT_PIXEL_FORMAT) if self.settings.get('format', params.DEFAULT_PIXEL_FORMAT) is not None else params.DEFAULT_PIXEL_FORMAT,
            'SensorBitDepth': 'Bpp12',
            'AdaptiveNoiseSuppressionFactor': 1.0,
            'AcquisitionFrameRateEnable': True,
            'AcquisitionFrameRate': self.settings.get('max_framerate', params.DEFAULT_MAX_FRAMERATE) if self.settings.get('max_framerate', params.DEFAULT_MAX_FRAMERATE) is not None else params.DEFAULT_MAX_FRAMERATE,
        }
        self.set_values(values_dict)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cam is not None:
            self.cam.__exit__(exc_type, exc_val, exc_tb)
        self._vmb_instance.__exit__(exc_type, exc_val, exc_tb)
        self._vmb = None
        self.cam = None

    def _open_camera(self) -> vmbpy.Camera:
        """Fetch the camera handle from the already-open VmbSystem."""
        if self.camera_id:
            try:
                cam = self._vmb.get_camera_by_id(self.camera_id)
                self.logger.info(f'Found camera {self.camera_id}')
                return cam
            except vmbpy.VmbCameraError:
                self.logger.error(f"Failed to access Camera '{self.camera_id}'. Abort.")
                sys.exit(1)
        else:
            cams = self._vmb.get_all_cameras()
            if not cams:
                self.logger.error('No cameras accessible. Abort.')
                sys.exit(1)
            return cams[0]
 
    def _read_all_features(self):
        """Read all feature values from the already-open camera handle."""
        self.logger.info('Downloading camera features...')
        features = {}
        for feat in self.cam.get_all_features():
            name = feat.get_name()
            features[name] = {
                'display_name':   feat.get_display_name(),
                'tooltip':        feat.get_tooltip(),
                'description':    feat.get_description(),
                'sfnc_namespace': feat.get_sfnc_namespace(),
            }
            try:
                features[name]['value'] = feat.get()
            except (AttributeError, vmbpy.VmbFeatureError):
                features[name]['value'] = None
        self.logger.info('Finished downloading camera features.')
        return features

    def get_all_features(self):
        """Refresh the feature cache from the camera and return it."""
        self.features = self._read_all_features()
        return self.features
    
    def log_all_features(self, filename: str = 'camera_features.txt'):
        filename = os.path.join(self.output_path, filename) if self.output_path else filename
        self.logger.info(f'Saving camera features to file: {filename}')
        if self.features is None:
            self.features = self.get_all_features()
        with open(filename, 'w') as f:
            for feat_name in self.features.keys():
                f.write(f"{feat_name}\n")
                f.write(f"    Display name   : {self.features[feat_name]['display_name']}\n")
                f.write(f"    Tooltip        : {self.features[feat_name]['tooltip']}\n")
                f.write(f"    Description    : {self.features[feat_name]['description']}\n")
                f.write(f"    SFNC Namespace : {self.features[feat_name]['sfnc_namespace']}\n")
                f.write(f"    Value          : {str(self.features[feat_name]['value'])}\n\n")
        self.logger.info(f'Camera features saved to {filename}')

    def get_value(self, feature_name):
        if self.features is None:
            self.features = self.get_all_features()
        return self.features.get(feature_name, {}).get('value', None)
    
    def set_values(self, feature_dict):
        """Set a dict of {feature_name: value} on the already-open camera."""
        for feature_name, value in feature_dict.items():
            try:
                feat = self.cam.get_feature_by_name(feature_name)
                feat.set(value)
                self.logger.info(f'Set {feature_name} -> {value}')
                if self.features is not None and feature_name in self.features:
                    self.features[feature_name]['value'] = value
            except (AttributeError, vmbpy.VmbFeatureError) as e:
                self.logger.error(f'Failed to set feature {feature_name} to {value}: {e}')

    def get_frame(self):
        try:
            return self.cam.get_frame()
        except vmbpy.VmbCameraError as e:
            self.logger.error(f'Failed to get frame: {e}')
            return None

    def start_continuous_streaming(self, callback, buffer_count: int = 10,
                                    timing: bool = False,
                                    monitored_queue: Queue = None):
        """Start streaming indefinitely, calling callback(frame) for every frame.

        Args:
            callback:         called with the VmbPy Frame object for every complete
                              frame. Runs on the SDK thread - must be fast.
            buffer_count:     number of SDK DMA buffers to pre-announce.
            timing:           if True, record per-frame timing data. Retrieve
                              results with get_streaming_stats() after stopping.
            monitored_queue:  a Queue used by the writer thread. If provided,
                              its size is sampled at every frame arrival so you
                              can see whether the writer is keeping up.

        Use timed_write() as a context manager in the writer thread to record
        individual disk-write durations:

            def writer(q):
                while True:
                    frame = q.get()
                    if frame is None:
                        break
                    with camera.timed_write():
                        cv2.imwrite('frame.tiff', frame.as_numpy_ndarray())

            camera.start_continuous_streaming(lambda f: q.put(f),
                                              timing=True,
                                              monitored_queue=q)
        """
        # reset stats storage
        self._streaming_stats = {
            'host_timestamps_s':  [],
            'camera_timestamps':  [],
            'queue_sizes':        [],
            'write_durations_s':  [],
            'write_timestamps_s': [],
            'cpu_usage_percent':  [],
            'ram_usage_percent':  [],
        }
        self._timing_enabled   = timing
        self._monitored_queue  = monitored_queue

        def handler(cam, stream, frame):
            if frame.get_status() == vmbpy.FrameStatus.Complete:
                if self._timing_enabled:
                    self._streaming_stats['host_timestamps_s'].append(time.time_ns())
                    try:
                        self._streaming_stats['camera_timestamps'].append(frame.get_timestamp())
                    except Exception:
                        self._streaming_stats['camera_timestamps'].append(None)
                    if self._monitored_queue is not None:
                        self._streaming_stats['queue_sizes'].append(
                            self._monitored_queue.qsize())
                    self._streaming_stats['cpu_usage_percent'].append(
                        psutil.cpu_percent())
                    self._streaming_stats['ram_usage_percent'].append(
                        psutil.virtual_memory().percent)
                callback(frame)
            cam.queue_frame(frame)

        self.cam.start_streaming(handler=handler, buffer_count=buffer_count)
        self.logger.info('Continuous streaming started.')

    def stop_continuous_streaming(self):
        """Stop a stream started with start_continuous_streaming()."""
        self.cam.stop_streaming()
        self.logger.info('Continuous streaming stopped.')

    def start_acquisition(self):
        """Start acquisition on the already-open camera."""
        logging.info('Starting acquisition...')

        self.dispatcher = threading.Thread(
            target=self.dispatcher_thread,
            args=(self.frame_queue, self.output_path, self.writing_threads),
            daemon=True,
            name='dispatcher',
        )
        self.dispatcher.start()

        self.start_continuous_streaming(
            callback=lambda f: self.frame_queue.put((f, self._streaming_stats['host_timestamps_s'][-1], self._streaming_stats['camera_timestamps'][-1])),
            timing=True,
            monitored_queue=self.frame_queue,
        )

    def stop_acquisition(self):
        self.stop_continuous_streaming()
        time.sleep(0.5)              # give the dispatcher a moment to finish processing frames
        self.frame_queue.put(None)   # poison pill: stop the dispatcher
        self.dispatcher.join()  
        
    @contextmanager
    def timed_write(self):
        """Context manager to time a single disk-write operation.

        Use inside the writer thread when timing=True was passed to
        start_continuous_streaming(). Records the wall-clock start time
        and duration of each write block in the streaming stats.

        Example:
            with camera.timed_write():
                cv2.imwrite('frame.tiff', arr)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - t0
            if hasattr(self, '_streaming_stats'):
                self._streaming_stats['write_timestamps_s'].append(t0)
                self._streaming_stats['write_durations_s'].append(duration)

    def _write_frame(self, arr, path):
        """Write a single frame to disk, recording duration via camera.timed_write()."""
        with self.timed_write():
            with open(path, 'wb') as f:
                f.write(arr)


    def dispatcher_thread(self, frame_queue, output_path, n_workers):
        """Pull frames from the queue and dispatch writes to the thread pool.

        Each frame's numpy array is extracted here (in the dispatcher thread)
        before submission so the SDK frame buffer is not held by a worker.
        Workers only touch the already-copied numpy array.
        """
        futures = []
        i = 0
        with ThreadPoolExecutor(max_workers=n_workers,
                                thread_name_prefix='nv-writer') as executor:
            while True:
                item = frame_queue.get()
                if item is None:   # poison pill
                    break
                frame, host_timestamp, camera_timestamp = item
                arr  = frame.as_numpy_ndarray().tobytes()
                path = f'{output_path}/frame_{i:06d}.raw'
                self.timestamp_logger.info(f'host_time_ns={host_timestamp} camera_time={camera_timestamp} frame={i:06d}')
                futures.append(executor.submit(self._write_frame, arr, path))
                i += 1
            # executor.__exit__ waits for all submitted futures before returning

        # propagate any worker exceptions
        for fut in futures:
            exc = fut.exception()
            if exc:
                raise exc

    def get_streaming_stats(self) -> dict:
        """Return derived statistics from the last continuous streaming run.

        Call this after stop_continuous_streaming().

        Returns a dict with:
          Raw data:
            host_timestamps_s   per-frame host timestamps (perf_counter, s)
            camera_timestamps   per-frame camera hardware ticks
            queue_sizes         queue depth sampled at each frame arrival
            write_durations_s   duration of each timed_write block (s)
            write_timestamps_s  host timestamp of each write start

          Derived - acquisition:
            n_frames            total frames received
            intervals_s         n-1 inter-frame intervals (s)
            mean_fps / std_fps  mean and std of instantaneous fps
            total_duration_s    time from first to last frame

          Derived - queue:
            mean_queue_size / max_queue_size
            queue_overflow_ratio  fraction of frames where queue was non-empty
                                  (writer not keeping up)

          Derived - disk writes:
            n_writes
            mean_write_s / std_write_s / max_write_s   write latency stats
            mean_write_fps  average write throughput (writes per second)
        """
        if not hasattr(self, '_streaming_stats'):
            self.logger.warning('No streaming stats available. '
                           'Run start_continuous_streaming with timing=True first.')
            return {}

        s = self._streaming_stats
        stats = dict(s)  # include raw data

        # acquisition timing
        ts = s['host_timestamps_s']
        stats['n_frames'] = len(ts)
        if len(ts) > 1:
            intervals = np.diff(ts)
            fps_vals  = 1.0 / intervals[intervals > 0]
            stats['intervals_s']      = intervals
            stats['mean_fps']         = float(np.mean(fps_vals))
            stats['median_fps']       = float(np.median(fps_vals))
            stats['std_fps']          = float(np.std(fps_vals))
            stats['total_duration_s'] = float(ts[-1] - ts[0])
        else:
            stats['intervals_s']      = np.array([])
            stats['mean_fps']         = float('nan')
            stats['median_fps']       = float('nan')
            stats['std_fps']          = float('nan')
            stats['total_duration_s'] = float('nan')

        # queue stats
        qs = s['queue_sizes']
        if qs:
            qs_arr = np.array(qs)
            stats['mean_queue_size']      = float(np.mean(qs_arr))
            stats['median_queue_size']    = float(np.median(qs_arr))
            stats['max_queue_size']       = int(np.max(qs_arr))
            stats['queue_overflow_ratio'] = float(np.mean(qs_arr > 0))
        else:
            stats['mean_queue_size']      = float('nan')
            stats['median_queue_size']    = float('nan')
            stats['max_queue_size']       = 0
            stats['queue_overflow_ratio'] = float('nan')

        # write stats
        wd = s['write_durations_s']
        if wd:
            wd_arr = np.array(wd)
            stats['n_writes']        = len(wd)
            stats['mean_write_s']    = float(np.mean(wd_arr))
            stats['median_write_s']  = float(np.median(wd_arr))
            stats['std_write_s']     = float(np.std(wd_arr))
            stats['max_write_s']     = float(np.max(wd_arr))
            wt = np.array(s['write_timestamps_s'])
            if len(wt) > 1:
                stats['mean_write_fps'] = float(len(wt) / (wt[-1] - wt[0] + wd_arr[-1]))
            else:
                stats['mean_write_fps'] = float('nan')
        else:
            stats['n_writes']        = 0
            stats['mean_write_s']    = float('nan')
            stats['median_write_s']  = float('nan')
            stats['std_write_s']     = float('nan')
            stats['max_write_s']     = float('nan')
            stats['mean_write_fps']  = float('nan')

        self.timestamp_logger.info(f"Streaming stats: {stats['n_frames']} frames")
        self.timestamp_logger.info(f"fps            = {stats['mean_fps']:.2f}±{stats['std_fps']:.2f}")
        self.timestamp_logger.info(f"median fps     = {stats['median_fps']:.2f}")
        self.timestamp_logger.info(f"writes         = {stats['n_writes']}")
        self.timestamp_logger.info(f"mean_write     = {stats['mean_write_s']*1e3:.1f} ms")
        self.timestamp_logger.info(f"median_write   = {stats['median_write_s']*1e3:.1f} ms")
        self.timestamp_logger.info(f"queue_overflow = {stats['queue_overflow_ratio']:.1%}")
        

        # convert all lists to np.arrays
        for key in ['host_timestamps_s', 'camera_timestamps', 'intervals_s',
                    'queue_sizes', 'write_durations_s', 'write_timestamps_s']:
            stats[key] = np.array(stats[key])
        return stats