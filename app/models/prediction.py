"""
DermaCortex AI - Prediction Model
MongoDB document schema for skin disease predictions
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class PredictionResult(BaseModel):
    """Single prediction result"""
    disease: str
    confidence: float = Field(..., ge=0, le=1)
    description: str
    recommendation: str
    ingredients: Optional[List[str]] = None


class Prediction(BaseModel):
    """Prediction document model"""
    
    # User reference
    user_id: str = Field(...)
    
    # Image information
    image_url: str
    image_filename: Optional[str] = None
    
    # Prediction results
    predictions: List[PredictionResult] = Field(default_factory=list)
    top_prediction: Optional[PredictionResult] = None
    confidence_score: float = Field(..., ge=0, le=1)
    
    # Additional info
    body_part: Optional[str] = None
    skin_type: Optional[str] = None
    symptoms: Optional[List[str]] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Status
    status: str = Field(default="completed")  # pending, processing, completed, failed
    error_message: Optional[str] = None
    
    # Metadata
    model_version: str = Field(default="1.0.0")
    processing_time_ms: Optional[int] = None
    
    model_config = ConfigDict(protected_namespaces=())


class PredictionResponse(BaseModel):
    """Prediction response model"""
    
    id: str = Field(..., alias="_id")
    user_id: str
    image_url: str
    predictions: List[PredictionResult]
    top_prediction: Optional[PredictionResult] = None
    confidence_score: float
    body_part: Optional[str] = None
    skin_type: Optional[str] = None
    symptoms: Optional[List[str]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    model_version: str
    processing_time_ms: Optional[int] = None
    
    model_config = ConfigDict(protected_namespaces=())


class PredictionCreate(BaseModel):
    """Prediction creation request model"""
    image_url: str
    image_filename: Optional[str] = None
    body_part: Optional[str] = None
    skin_type: Optional[str] = None
    symptoms: Optional[List[str]] = None


class PredictionListResponse(BaseModel):
    """Prediction list response model"""
    predictions: List[PredictionResponse]
    total: int
    page: int
    page_size: int
    has_more: bool

    model_config = ConfigDict(protected_namespaces=())
