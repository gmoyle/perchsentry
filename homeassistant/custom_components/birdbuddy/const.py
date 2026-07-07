"""Constants for the BirdBuddy integration."""

DOMAIN = "birdbuddy"

CONF_HOST = "host"
CONF_PORT = "port"

DEFAULT_HOST = "birdbuddy.local"
DEFAULT_PORT = 8080
DEFAULT_SCAN_INTERVAL = 5  # seconds between /api/ha polls

# Events fired on the HA bus when a new detection lands — automate notifications
# off these.
EVENT_BIRD_DETECTED = "birdbuddy_bird_detected"
EVENT_ANIMAL_DETECTED = "birdbuddy_animal_detected"

# How long (seconds) a detection keeps its binary sensor "on" afterward.
DETECTION_HOLD = 60
