import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import utils
import config
import image_utils
from typing import Optional
import base64
import io

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="API for Air Quality Index prediction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models
model = utils.load_model()
image_model = image_utils.load_image_model()


class PredictionInput(BaseModel):
    pm25: float
    pm10: float
    no: float
    no2: float
    nox: float
    nh3: float
    co: float
    so2: float
    o3: float
    benzene: float
    toluene: float


class PredictionOutput(BaseModel):
    aqi: float
    category: str
    recommendation: str


class ImagePredictionOutput(BaseModel):
    pm25: float
    category: str
    recommendation: str


class Base64ImageInput(BaseModel):
    image: str  # base64 encoded image
    location: Optional[str] = None
    datetime: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "AQI Prediction API is running"}


@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "image_model_loaded": image_model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess input data
    features = utils.preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(features)[0][0]
    
    # Determine AQI category
    category = utils.get_aqi_category(prediction)
    
    # Get health recommendations
    recommendation = utils.get_health_recommendations(category)
    
    return PredictionOutput(
        aqi=float(prediction), 
        category=category,
        recommendation=recommendation
    )


@app.post("/predict/image", response_model=ImagePredictionOutput)
async def predict_from_image(
    image: UploadFile = File(...),
    location: Optional[str] = Form(None),
    datetime: Optional[str] = Form(None)
):
    """
    Predict PM2.5 from an uploaded image.
    
    - **image**: Image file to analyze
    - **location**: Optional location where the image was taken
    - **datetime**: Optional date and time when the image was taken
    """
    if image_model is None:
        raise HTTPException(status_code=500, detail="Image model not loaded")
    
    # Check file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # Read the image file
        contents = await image.read()
        
        # Predict PM2.5 from image
        pm25_prediction = image_utils.predict_pm25_from_image(image_model, contents)
        
        # Determine category based on PM2.5 value
        category = utils.get_aqi_category(pm25_prediction)
        
        # Get health recommendations
        recommendation = utils.get_health_recommendations(category)
        
        return ImagePredictionOutput(
            pm25=pm25_prediction,
            category=category,
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict/base64", response_model=ImagePredictionOutput)
async def predict_from_base64(input_data: Base64ImageInput):
    """
    Predict PM2.5 from a base64 encoded image.
    
    - **image**: Base64 encoded image string
    - **location**: Optional location where the image was taken
    - **datetime**: Optional date and time when the image was taken
    """
    if image_model is None:
        raise HTTPException(status_code=500, detail="Image model not loaded")
    
    try:
        # Clean and validate base64 string
        base64_str = input_data.image
        # Remove potential whitespace, newlines and other invalid characters
        base64_str = ''.join(base64_str.split())
        
        # Remove potential data URI prefix
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
            
        # Add padding if necessary
        padding = len(base64_str) % 4
        if padding:
            base64_str += '=' * (4 - padding)
            
        try:
            image_bytes = base64.b64decode(base64_str)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid base64 image data: {str(e)}"
            )
            
        # Verify we have actual image data
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty image data"
            )
        
        # Predict PM2.5 from image
        pm25_prediction = image_utils.predict_pm25_from_image(image_model, image_bytes)
        
        # Determine category based on PM2.5 value
        category = utils.get_aqi_category(pm25_prediction)
        
        # Get health recommendations
        recommendation = utils.get_health_recommendations(category)
        
        return ImagePredictionOutput(
            pm25=float(pm25_prediction),
            category=category,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def model_info():
    """Get information about the loaded models"""
    
    # Get information about the numeric input model
    numeric_model_info = {}
    if model:
        numeric_model_info = {
            "input_shape": [x.shape.as_list() for x in model.inputs],
            "output_shape": [x.shape.as_list() for x in model.outputs],
            "layers": len(model.layers),
        }
    
    # Get information about the image model
    image_model_info = {}
    if image_model:
        image_model_info = {
            "input_shape": [x.shape.as_list() for x in image_model.inputs],
            "output_shape": [x.shape.as_list() for x in image_model.outputs],
            "layers": len(image_model.layers),
        }
    
    return {
        "numeric_model": numeric_model_info,
        "image_model": image_model_info
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("app:app", host=config.APP_HOST, port=config.APP_PORT, reload=config.DEBUG)