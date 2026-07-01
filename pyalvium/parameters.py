# frame timestamps logging defaults
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '[%(asctime)s.%(msecs)03d] (%(levelname)s): %(message)s'
LOGGING_DATEFMT = '%Y-%m-%d %H:%M:%S'
LOGGING_FILENAME = 'frame_timestamps.log'
VERBOSE = False

# path defaults
DEFAULT_OUTPUT_PATH = 'home/polocalc/data/unlabeled_camera_data'

# camera defaults
CAMERA_ID                = "DEV_1AB22C0707F3"
DEFAULT_WRITING_THREADS  = 1
DEFAULT_EXPOSURE_TIME_US = 10000       # microseconds
DEFAULT_GAIN             = 30          # dB
DEFAULT_MAX_FRAMERATE    = 5           # frames per second
DEFAULT_PIXEL_FORMAT     = 'BayerRG8'
DEFAULT_ROI_X_OFFSET     = 0
DEFAULT_ROI_Y_OFFSET     = 0
DEFAULT_REVERSE_X        = False
DEFAULT_REVERSE_Y        = False
DEFAULT_BINNING_SIZE_X   = 1
DEFAULT_BINNING_SIZE_Y   = 1
DEFAULT_SHARPNESS_FACTOR = 0.0
DEFAULT_SATURATION       = 1.0