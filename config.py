import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "./model.h5")
# Image model settings (can be the same as the main model or different)
IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", MODEL_PATH)

# API settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "").split(",")

# Image processing settings
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB default
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "200"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "120"))

# AQI categories thresholds
AQI_GOOD = 50
AQI_MODERATE = 100
AQI_UNHEALTHY_SENSITIVE = 150
AQI_UNHEALTHY = 200
AQI_VERY_UNHEALTHY = 300

# PM2.5 categories thresholds (μg/m³)
PM25_GOOD = 12
PM25_MODERATE = 35.4
PM25_UNHEALTHY_SENSITIVE = 55.4
PM25_UNHEALTHY = 150.4
PM25_VERY_UNHEALTHY = 250.4

# AQI recommendations
AQI_RECOMMENDATIONS = {
    "Good": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable; however, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
    "Unhealthy": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
    "Hazardous": "Health warnings of emergency conditions. The entire population is more likely to be affected."
}