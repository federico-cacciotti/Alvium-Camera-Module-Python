from vmbpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Camera:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._vmb = None
        self._cam = None
        self.exposure_time = None
        self.gain = None
        self.image_width = None
        self.image_height = None
        self._last_frame = None

    def open(self):
        self._vmb = VmbSystem.get_instance()
        self._vmb.__enter__()
        self._cam = self._vmb.get_camera_by_id(self.camera_id)
        self._cam.__enter__()

    def close(self):
        if self._cam:
            self._cam.__exit__(None, None, None)
        if self._vmb:
            self._vmb.__exit__(None, None, None)

    def set_pixel_format(self, pixel_format: PixelFormat):
        self._cam.set_pixel_format(pixel_format)
        print(f"Pixel format set to {pixel_format}")

    def set_size(self, width: int, height: int):
        self._cam.get_feature_by_name("Width").set(width)
        self.image_width = self._cam.get_feature_by_name("Width").get()
        print(f"Image width set to {self.image_width}")
        self._cam.get_feature_by_name("Height").set(height)
        self.image_height = self._cam.get_feature_by_name("Height").get()
        print(f"Image height set to {self.image_height}")

    def configure_raw(self, exposure_time: float, gain: float):
        # set exposure time
        self._cam.get_feature_by_name("ExposureAuto").set("Off")
        self._cam.get_feature_by_name("ExposureTime").set(exposure_time)
        self.exposure_time = self._cam.get_feature_by_name("ExposureTime").get()
        print(f"Exposure time set to {self.exposure_time} microseconds")

        # set gain
        self._cam.get_feature_by_name("GainAuto").set("Off")
        self._cam.get_feature_by_name("Gain").set(gain)
        self.gain = self._cam.get_feature_by_name("Gain").get()
        print(f"Gain set to {self.gain}")

        # disagble white balance and set to neutral
        self._cam.get_feature_by_name("BalanceWhiteAuto").set("Off")
        self._cam.get_feature_by_name("BalanceRatioSelector").set("Red") # only red available
        self._cam.get_feature_by_name("BalanceRatio").set(1.0)
        print("White balance set to neutral")

        # set gamma to one
        try:
            self._cam.get_feature_by_name("Gamma").set(1.0)
            print("Gamma set to 1.0")
        except:
            print("Gamma feature not available")
            pass

        # disable sharpening
        try:
            self._cam.get_feature_by_name("Sharpness").set(0)
            print("Sharpness set to 0")
        except:
            print("Sharpness feature not available")
            pass

        # disable color transformation
        try:
            self._cam.get_feature_by_name("ColorTransformationEnable").set(False)
            print("Color transformation disabled")
        except:
            print("Color transformation feature not available")
            pass


    def get_frame(self, timeout: int = 3000):
        try:
            frame = self._cam.get_frame(timeout_ms=timeout)
            print("Frame acquired")
            return frame
        except TimeoutError:
            print("Timeout error while acquiring frame")
            return None
        
    def _frame_callback(self, cam, stream, frame):
        if frame.get_status() == FrameStatus.Complete:
            self._last_frame = frame.as_numpy_ndarray().copy()
            if len(self._last_frame.shape) == 3:
                self._last_frame = self._last_frame[:, :, ::-1]  # BGR → RGB
        cam.queue_frame(frame)

    def stream_live(self, grid: bool = False):
        streams = self._cam.get_streams()
        if not streams:
            print("No streams available")
            return

        fig, ax = plt.subplots()
        ax.axis("off")
        img_plot = None

        streams[0].start_streaming(handler=self._frame_callback, buffer_count=10)
        try:
            while plt.fignum_exists(fig.number):
                if self._last_frame is not None:
                    if img_plot is None:
                        img_plot = ax.imshow(self._last_frame, cmap="gray")
                        if grid:
                            ax.axvline(x=self.image_width // 2, color="red", linestyle="solid")
                            ax.axhline(y=self.image_height // 2, color="red", linestyle="solid")
                    else:
                        img_plot.set_data(self._last_frame)
                        img_plot.set_clim(vmin=self._last_frame.min(),
                                        vmax=self._last_frame.max())
                plt.pause(0.05)
        finally:
            streams[0].stop_streaming()
            plt.close(fig)