from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from chat import get_gemini_response, ChatRequest

app = FastAPI(
    title="Crop Disease Detection API",
    description="API for detecting diseases in crop images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
MODEL = None
def load_model():
    global MODEL
    try:
        MODEL = tf.keras.models.load_model('model/crop_disease_model.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Class names in correct order
CLASS_NAMES = [
    'Corn___Common_Rust',
    'Corn___Gray_Leaf_Spot',
    'Corn___Healthy',
    'Corn___Leaf_Blight',
    'Invalid',
    'Potato___Early_Blight',
    'Potato___Healthy',
    'Potato___Late_Blight',
    'Rice___Brown_Spot',
    'Rice___Healthy',
    'Rice___Hispa',
    'Rice___Leaf_Blast',
    'Wheat___Brown_Rust',
    'Wheat___Healthy',
    'Wheat___Yellow_Rust'
]

async def predict_image(image: Image.Image):
    """Make prediction on a single image"""
    try:
        # Preprocess the image
        img = image.resize((160, 160))  # Resize to match training size
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        
        # Make prediction
        predictions = MODEL.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index] * 100)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx] * 100)
            }
            for idx in top_3_indices
        ]
        
        # Get crop type and condition
        predicted_class = CLASS_NAMES[predicted_class_index]
        crop_type, condition = predicted_class.split('___')
        
        return {
            'status': 'success',
            'predicted_class': predicted_class,
            'confidence': confidence,
            'crop_type': crop_type,
            'condition': condition,
            'top_3_predictions': top_3_predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():

    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Crop Disease Detection API",
        "usage": "POST an image to /predict endpoint"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict crop disease from uploaded image
    
    Parameters:
    - file: Image file to analyze
    
    Returns:
    - Prediction results including disease class and confidence
    """
    try:
        # Verify file extension
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a PNG or JPG image."
            )
        # Read and verify image
        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload a valid image."
            )
        # Make prediction
        result = await predict_image(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat_with_expert(chat_request: ChatRequest):
#     """
#     Chat endpoint for disease-specific questions
    
#     Parameters:
#     - disease_info: Disease prediction information
#     - question: User's question about the disease
    
#     Returns:
#     - AI response focused on the specific disease
#     """
#     try:
#         response = await get_gemini_response(
#             chat_request.disease_info,
#             chat_request.question
#         )
        
#         return JSONResponse(content={
#             "status": "success",
#             "disease": chat_request.disease_info["predicted_class"],
#             "question": chat_request.question,
#             "response": response
#         })
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/chat")
async def chat_with_expert(chat_request: ChatRequest):
    """
    Chat endpoint for disease-specific questions
    
    Parameters:
    - disease_info: Disease prediction information
    - question: User's question about the disease
    - chat_history: Optional previous chat history
    
    Returns:
    - AI response focused on the specific disease and updated chat history
    """
    try:
        chat_response = await get_gemini_response(
            chat_request.disease_info,
            chat_request.question,
            chat_request.chat_history
        )
        
        return JSONResponse(content={
            "status": "success",
            "disease": chat_request.disease_info["predicted_class"],
            "question": chat_request.question,
            "response": chat_response['response'],
            "chat_history": chat_response['history']
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }



# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    