import sys
import numpy as np
import time
import threading
import logging
from contextlib import contextmanager
from queue import Queue, Empty
import vmbpy
import psutil
import os

import parameters as params

# VmbPy logs every internal error via a logger named 'vmbpyLog'. With propagate=True
# (Python default) those records bubble up to the root logger and pollute the output.
# We only want our own application-level logs, so disable propagation for vmbpy's logger.
logging.getLogger('vmbpyLog').propagate = False

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

    def __init__(self, camera_id=None, output_dir=None):

        logging_filename = os.path.join(output_dir, params.LOGGING_FILENAME) if output_dir else params.LOGGING_FILENAME

        # configure logging
        logging.basicConfig(level=getattr(logging, params.LOGGING_LEVEL), 
                            format=params.LOGGING_FORMAT, 
                            datefmt=params.LOGGING_DATEFMT,
                            handlers=[logging.StreamHandler(sys.stdout),
                                    logging.FileHandler(logging_filename)])
        self.logger = logging.getLogger(__name__)

        self.camera_id = camera_id
        self.output_dir = output_dir
        self._vmb_instance = vmbpy.VmbSystem.get_instance()
        self._vmb = None   # set in __enter__
        self.cam = None    # set in __enter__
        self.features = None

    def __enter__(self):
        self._vmb = self._vmb_instance.__enter__()
        self.cam = self._open_camera()
        self.cam.__enter__()
        self.logger.info(f'Camera opened: {self.cam.get_id()} - {self.cam.get_model()}')
        self.features = self._read_all_features()
        self._run_startup_checks()
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
                self.logger.debug(f'Found camera {self.camera_id}')
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

    def _run_startup_checks(self):
        # check if shutter mode is set to global
        # EnumEntry.__str__ decodes a null-padded C char array, strip null bytes before comparing
        shutter_mode = self.get_value('SensorShutterMode')
        if str(shutter_mode).rstrip('\x00') != 'GlobalShutter':
            self.logger.warning(f"Camera shutter mode is set to '{shutter_mode}', but 'GlobalShutter' is recommended")

        # check if the sensor is cropped
        width = self.get_value('Width')
        height = self.get_value('Height')
        width_max = self.get_value('WidthMax')
        height_max = self.get_value('HeightMax')
        offset_x = self.get_value('OffsetX')
        offset_y = self.get_value('OffsetY')
        if None not in (width, height, width_max, height_max):
            if width != width_max or height != height_max:
                self.logger.warning(f"Camera sensor is cropped: Width={width}/{width_max}, Height={height}/{height_max}, OffsetX={offset_x}, OffsetY={offset_y}")
            
    def _read_all_features(self):
        """Read all feature values from the already-open camera handle."""
        self.logger.debug('Downloading camera features...')
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
        self.logger.debug('Finished downloading camera features.')
        return features

    def get_all_features(self):
        """Refresh the feature cache from the camera and return it."""
        self.features = self._read_all_features()
        return self.features
    
    def log_all_features(self, filename: str = 'camera_features.txt'):
        filename = os.path.join(self.output_dir, filename) if self.output_dir else filename
        self.logger.debug(f'Saving camera features to file: {filename}')
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
        self.logger.debug(f'Camera features saved to {filename}')

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

    def get_frames(self, n: int, fps: float = None, timing: bool = False) -> list:
        """Acquire n frames at maximum speed using asynchronous streaming.

        Uses a pre-announced buffer pool so the camera runs continuously with
        no gap between frames, unlike calling get_frame() in a loop which
        stalls acquisition between each synchronous round-trip.

        Args:
            n:      number of frames to acquire.
            fps:    if not None, set the camera to this framerate (fps).
            timing: if True, also return a dict with timing statistics.
                    The dict contains:
                      'host_timestamps_s'  : list of n host-side timestamps
                                             (time.perf_counter(), seconds)
                                             recorded the moment each frame
                                             arrives in the callback.
                      'camera_timestamps'  : list of n camera hardware
                                             timestamps (ticks, from
                                             frame.get_timestamp()). Useful
                                             to measure the true inter-frame
                                             period independently of host
                                             scheduling jitter.
                      'intervals_s'        : list of n-1 host-side inter-frame
                                             intervals in seconds.
                      'mean_fps'           : mean fps from host intervals.
                      'std_fps'            : standard deviation of fps.

        Returns:
            frames            if timing=False (default)
            (frames, timing)  if timing=True
        """
        frames = []
        host_timestamps = []
        camera_timestamps = []
        done = threading.Event()

        if fps is not None:
            self.set_values({'AcquisitionFrameRateEnable': True,
                             'AcquisitionFrameRate': fps})
        else:
            self.set_values({'AcquisitionFrameRateEnable': False})

        def handler(cam, stream, frame):
            if frame.get_status() == vmbpy.FrameStatus.Complete and len(frames) < n:
                if timing:
                    host_timestamps.append(time.perf_counter())
                    try:
                        camera_timestamps.append(frame.get_timestamp())
                    except Exception:
                        camera_timestamps.append(None)
                frames.append(frame)
                if len(frames) == n:
                    done.set()
            cam.queue_frame(frame)

        buffer_count = min(max(n, 10), 50)
        self.cam.start_streaming(handler=handler, buffer_count=buffer_count)
        self.logger.info(f'Streaming started, acquiring {n} frames...')
        try:
            done.wait()
        finally:
            self.cam.stop_streaming()

        self.logger.info(f'Acquired {len(frames)} frames.')

        if not timing:
            return frames

        intervals = np.array([host_timestamps[i+1] - host_timestamps[i]
                     for i in range(len(host_timestamps) - 1)])
        if intervals.size > 0:
            fps_values = [1.0 / dt for dt in intervals if dt > 0]
            mean_fps = sum(fps_values) / len(fps_values)
            variance = sum((f - mean_fps) ** 2 for f in fps_values) / len(fps_values)
            std_fps = variance ** 0.5
        else:
            mean_fps = std_fps = float('nan')

        self.logger.info(f'Timing: mean fps={mean_fps:.2f}, std={std_fps:.2f}')

        timing_info = {
            'host_timestamps_s': host_timestamps,
            'camera_timestamps': camera_timestamps,
            'intervals_s':       intervals,
            'mean_fps':          mean_fps,
            'std_fps':           std_fps,
        }
        return frames, timing_info

    def start_continuous_streaming(self, callback, buffer_count: int = 10,
                                    timing: bool = False,
                                    monitored_queue: Queue = None):
        """Start streaming indefinitely, calling callback(frame) for every frame.

        Args:
            callback:         called with the VmbPy Frame object for every complete
                              frame. Runs on the SDK thread — must be fast.
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
                    self._streaming_stats['host_timestamps_s'].append(time.perf_counter())
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

        self.logger.info(f"Streaming stats: {stats['n_frames']} frames")
        self.logger.info(f"fps            = {stats['mean_fps']:.2f}±{stats['std_fps']:.2f}")
        self.logger.info(f"median fps     = {stats['median_fps']:.2f}")
        self.logger.info(f"writes         = {stats['n_writes']}")
        self.logger.info(f"mean_write     = {stats['mean_write_s']*1e3:.1f} ms")
        self.logger.info(f"median_write   = {stats['median_write_s']*1e3:.1f} ms")
        self.logger.info(f"queue_overflow = {stats['queue_overflow_ratio']:.1%}")
        

        # convert all lists to np.arrays
        for key in ['host_timestamps_s', 'camera_timestamps', 'intervals_s',
                    'queue_sizes', 'write_durations_s', 'write_timestamps_s']:
            stats[key] = np.array(stats[key])
        return stats