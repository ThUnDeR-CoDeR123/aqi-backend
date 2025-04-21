import numpy as np
import tensorflow as tf
import config

def load_model():
    """Load the TensorFlow model from the specified path."""
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        print(f"Model loaded successfully from {config.MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_input(input_data):
    """Convert input data to the format expected by the model."""
    features = np.array([[
        input_data.pm25,
        input_data.pm10, 
        input_data.no,
        input_data.no2,
        input_data.nox,
        input_data.nh3,
        input_data.co,
        input_data.so2,
        input_data.o3,
        input_data.benzene,
        input_data.toluene
    ]])
    return features


def get_aqi_category(aqi_value):
    """Categorize AQI value according to standard categories."""
    if aqi_value <= config.AQI_GOOD:
        return "Good"
    elif aqi_value <= config.AQI_MODERATE:
        return "Moderate"
    elif aqi_value <= config.AQI_UNHEALTHY_SENSITIVE:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= config.AQI_UNHEALTHY:
        return "Unhealthy"
    elif aqi_value <= config.AQI_VERY_UNHEALTHY:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_health_recommendations(category):
    """Get health recommendations based on AQI category."""
    return config.AQI_RECOMMENDATIONS.get(category, "No specific recommendations available.") 