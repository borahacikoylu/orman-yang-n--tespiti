# VIDEO settings: controls basic video processing cadence and resize target.
FRAME_SKIP = 2
INPUT_SIZE = 640

# OPENCV FILTER settings: configures HSV/motion filtering for candidate fire regions.
HSV_MIN_PIXEL_RATIO = 0.005
MOG2_HISTORY = 300
MOG2_THRESHOLD = 50
MOG2_DETECT_SHADOWS = False

# DETECTION threshold: shared confidence threshold used by CVDetector.
CONF_THRESHOLD = 0.25

# CV DETECTOR settings: defines HSV ranges and score parameters for OpenCV-only detection.
# HSV thresholds
HSV_FLAME_LOWER = [0, 150, 150]
HSV_FLAME_UPPER = [25, 255, 255]
HSV_EMBER_LOWER = [20, 150, 200]
HSV_EMBER_UPPER = [40, 255, 255]
HSV_SMOKE_LOWER = [0, 0, 100]
HSV_SMOKE_UPPER = [180, 55, 220]

# Contour filtering
MIN_CONTOUR_AREA = 150
MAX_SOLIDITY = 0.85
MAX_CIRCULARITY = 0.75

# Optical flow
FLOW_MAG_SCALE = 3.0
FLOW_VAR_SCALE = 2.0

# Score weights
WEIGHT_HSV = 0.35
WEIGHT_CONTOUR = 0.30
WEIGHT_FLOW = 0.20
WEIGHT_GROWTH = 0.15

# ALERT ENGINE settings: controls confidence/frame/area rules for alert escalation.
WATCH_CONF_MIN = 0.20
WARNING_CONF_MIN = 0.38
CRITICAL_CONF_MIN = 0.60
WATCH_MIN_FRAMES = 2
WARNING_MIN_FRAMES = 5
CRITICAL_MIN_FRAMES = 8
RESET_FRAMES = 10
SMOOTHING_WINDOW = 5
AREA_WATCH_MIN = 0.005
AREA_CRITICAL_MIN = 0.03
AREA_GROWTH_THRESHOLD = 0.20

# LOGGER settings: sets log storage path and alarm cooldown timing.
LOG_DIR = "logs"
ALARM_COOLDOWN_SEC = 30
