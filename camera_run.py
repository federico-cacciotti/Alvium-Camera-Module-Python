from Camera import Camera
import argparse
import time
import signal

# signals to catch to gracefully stop the acquisition
signals_to_catch = [signal.SIGINT, signal.SIGTERM]

parser = argparse.ArgumentParser(description='Alvium Camera Module - Python wrapper for official SDK')
parser.add_argument('--camera-id', type=str, default=None, help='Camera ID to use. If not specified, default will be used.')
parser.add_argument('--output-path', type=str, default=None, help='Directory to save frames. If not specified, default will be used.')
parser.add_argument('--exposure', type=int, default=None, help='Exposure time in microseconds. If not specified, default will be used.')
parser.add_argument('--gain', type=int, default=None, help='Gain in dB. If not specified, default will be used.')
parser.add_argument('--max-framerate', type=int, default=None, help='Framerate in frames per second. If not specified, default will be used.')
parser.add_argument('--format', type=str, default=None, help='Pixel format. If not specified, default will be used.')
parser.add_argument('--writing-threads', type=int, default=None, help='Number of writing threads. If not specified, default will be used.')
parser.add_argument('--saturation', type=float, default=None, help='Saturation factor. If not specified, default will be used.')
parser.add_argument('--sharpness', type=float, default=None, help='Sharpness factor. If not specified, default will be used.')
parser.add_argument('--duration', type=float, default=None, help='Duration of the acquisition in seconds. If not specified, acquisition will run until interrupted.'  )

def main():
    args = parser.parse_args()
    
    camera_id = args.camera_id
    output_path = args.output_path
    writing_threads = args.writing_threads
    duration = args.duration

    settings = {}
    settings['exposure']        = args.exposure
    settings['gain']            = args.gain
    settings['max_framerate']   = args.max_framerate
    settings['pixel_format']    = args.format
    settings['saturation']      = args.saturation
    settings['sharpness']       = args.sharpness

    with Camera(camera_id, output_path, writing_threads, settings) as camera:
        # log all camera features
        camera.log_all_features()

        # start acquisition until user interrupts
        camera.start_acquisition()

        # start acquisition for a specified duration if provided, otherwise wait for user interrupt
        if duration is not None:
            time.sleep(duration)
            camera.stop_acquisition()
        
        else:
            while True:
                time.sleep(0.2)  # wait for user interrupt
                for sig in signals_to_catch:
                    signal.signal(sig, camera.stop_acquisition())
                    return  # exit the program after stopping acquisition
            

if __name__ == '__main__':
    main()
