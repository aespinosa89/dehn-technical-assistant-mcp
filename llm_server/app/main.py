import os
import gc
import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from .models import load_model, get_model
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse
)
from .config import settings
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="DEHN Technical Assistant LLM Server",
    description="API server for LLM inference with vLLM",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    await load_model()

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "ok", "model": settings.MODEL_PATH}

# ChatCompletions API (OpenAI compatible)
@app.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse])
async def create_chat_completion(request: ChatCompletionRequest):
    model = await get_model()
    
    if request.stream:
        return StreamingResponse(
            model.generate_stream(request),
            media_type="text/event-stream"
        )
    else:
        return await model.generate(request)

# Embeddings API (OpenAI compatible)
@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingsRequest):
    model = await get_model()
    return await model.get_embeddings(request)

# Memory management endpoint
@app.post("/manage/gc")
async def manage_gc():
    """Force garbage collection to free up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "Memory cleared"}

# Model info
@app.get("/info")
async def model_info():
    """Get information about the loaded model"""
    model = await get_model()
    return {
        "model_name": model.model_name,
        "max_tokens": model.max_tokens,
        "temperature_range": {"min": 0.0, "max": 2.0, "default": 0.7},
        "gpu_info": torch.cuda.get_device_properties(0).__dict__ if torch.cuda.is_available() else None
    }
