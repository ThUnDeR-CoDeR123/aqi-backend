import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the image processing model
def load_image_model():
    """
    Load the model for image-based PM2.5 prediction.
    Uses the model file specified in config.
    """
    try:
        # Define the model architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(120, 200, 3)),
            tf.keras.layers.ZeroPadding2D(padding=(3, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Load weights from saved model
        model.load_weights(config.IMAGE_MODEL_PATH)
        logger.info(f"Image model loaded successfully from {config.IMAGE_MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        return None

def preprocess_image(image_bytes):
    """
    Preprocess the uploaded image for PM2.5 prediction.
    
    Args:
        image_bytes: Bytes of the uploaded image
        
    Returns:
        Preprocessed image as numpy array ready for model input
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to expected input size from config
    image = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def extract_image_features(img_array):
    """
    Extract relevant features from the image that might be useful for PM2.5 prediction.
    This is a simplified example - actual implementation would depend on your specific model.
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        Features extracted from the image
    """
    try:
        # Convert to grayscale for haze analysis
        if len(img_array.shape) == 4 and img_array.shape[3] == 3:
            img = img_array[0]  # Remove batch dimension
            
            # Convert to uint8 for OpenCV operations if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
                
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Calculate image statistics that might correlate with PM2.5
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            # Calculate image entropy (measure of texture)
            entropy = cv2.calcHist([gray], [0], None, [256], [0, 256])
            entropy = entropy[entropy > 0]
            entropy = -np.sum(entropy * np.log2(entropy)) if entropy.size > 0 else 0
            
            # Detect haze using gradient magnitude
            # Lower gradient magnitudes often indicate hazier images
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Create feature vector
            features = np.array([
                mean_intensity, 
                std_intensity, 
                contrast,
                entropy,
                avg_gradient
            ])
            
            logger.info(f"Extracted image features: {features}")
            return features.reshape(1, -1)  # Reshape to match model input
        
        logger.warning("Image array has unexpected shape")
        return np.zeros((1, 5))  # Default fallback with 5 features
    
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return np.zeros((1, 5))  # Default fallback

def get_pm25_category(pm25_value):
    """
    Categorize PM2.5 value according to standard PM2.5 categories.
    
    Args:
        pm25_value: PM2.5 concentration in μg/m³
        
    Returns:
        Category string
    """
    if pm25_value <= config.PM25_GOOD:
        return "Good"
    elif pm25_value <= config.PM25_MODERATE:
        return "Moderate"
    elif pm25_value <= config.PM25_UNHEALTHY_SENSITIVE:
        return "Unhealthy for Sensitive Groups"
    elif pm25_value <= config.PM25_UNHEALTHY:
        return "Unhealthy"
    elif pm25_value <= config.PM25_VERY_UNHEALTHY:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def predict_pm25_from_image(model, image_bytes):
    """
    Predict PM2.5 value from an image.
    
    Args:
        model: Loaded TensorFlow model
        image_bytes: Uploaded image bytes
        
    Returns:
        Predicted PM2.5 value
    """
    # Preprocess the image
    img_array = preprocess_image(image_bytes)
    
    try:
        # For direct image input to model
        if len(model.inputs[0].shape) >= 4 and model.inputs[0].shape[-1] == 3:
            # RGB input model - direct prediction
            logger.info("Using direct image prediction")
            prediction = model.predict(img_array)
        else:
            # Extract features if the model expects processed features
            logger.info("Using feature-based prediction")
            features = extract_image_features(img_array)
            prediction = model.predict(features)
        
        # Get the PM2.5 value from prediction
        if isinstance(prediction, list):
            pm25_value = prediction[0][0]  # Adjust based on your model output format
        else:
            pm25_value = prediction[0][0]
            
        logger.info(f"Predicted PM2.5 value: {pm25_value}")
        return float(pm25_value)
    
    except Exception as e:
        logger.error(f"Error predicting PM2.5: {e}")
        # Return a fallback value if prediction fails
        return 50.0  # Moderate level as fallback