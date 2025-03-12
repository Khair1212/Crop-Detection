from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    disease_info: dict
    question: str
    chat_history: Optional[List[dict]] = []

# Configure Gemini
GOOGLE_API_KEY = "AIzaSyABqWQd7qgm8ObR5DYdwmaYusGbv9mgQCE"
genai.configure(api_key=GOOGLE_API_KEY)

# Create model
model = genai.GenerativeModel('gemini-pro')

def create_chat_context(disease_info):
    """Create initial context for the chat"""
    return f"""
    You are a crop disease expert assistant. You have been provided with an image analysis of a crop disease.
    
    Analysis Results:
    - Detected Disease: {disease_info['predicted_class']}
    - Confidence: {disease_info['confidence']:.2f}%
    - Crop Type: {disease_info['crop_type']}
    - Condition: {disease_info['condition']}
    
    Using your extensive knowledge about crop diseases, especially about {disease_info['predicted_class']}, 
    provide accurate and helpful responses to questions. Focus only on the detected disease and relevant information.
    If asked about other diseases, politely redirect to the current disease context.
    """

async def get_gemini_response(disease_info, question, chat_history=None):
    """Get response from Gemini model using chat"""
    try:
        # Start new chat with context
        chat = model.start_chat(history=[])
        
        # Add initial context
        context = create_chat_context(disease_info)
        chat.send_message(context)
        
        # Add chat history if exists
        if chat_history:
            for msg in chat_history:
                chat.send_message(msg['content'])
        
        # Send user question and get response
        response = chat.send_message(question)
        
        return {
            'response': response.text,
            'history': chat.history
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# Set safety settings
# safety_settings = [
#     {
#         "category": genai.types.HarmCategory.HARASSMENT,
#         "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
#     },
#     {
#         "category": genai.types.HarmCategory.HATE_SPEECH,
#         "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
#     },
#     {
#         "category": genai.types.HarmCategory.SEXUALLY_EXPLICIT,
#         "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
#     },
#     {
#         "category": genai.types.HarmCategory.DANGEROUS_CONTENT,
#         "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
#     }
# ]

# # Create model with safety settings
# model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)