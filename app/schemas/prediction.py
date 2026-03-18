"""
DermaCortex AI - Prediction Schemas
Pydantic schemas for prediction-related requests and responses
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


# ==================== Request Schemas ====================

class PredictionCreateSchema(BaseModel):
    """Schema for creating a prediction"""
    image_url: str
    image_filename: Optional[str] = None
    body_part: Optional[str] = None
    skin_type: Optional[str] = None
    symptoms: Optional[List[str]] = None


class DiagnosisRequestSchema(BaseModel):
    """Schema for diagnosis request"""
    image_base64: str
    body_part: Optional[str] = None
    skin_type: Optional[str] = "normal"
    symptoms: Optional[List[str]] = None


# ==================== Response Schemas ====================

class PredictionItemSchema(BaseModel):
    """Schema for a single prediction item in the list"""
    disease: str
    confidence: int = Field(..., ge=0, le=100, description="Confidence as percentage (0-100)")


class TopPredictionSchema(BaseModel):
    """Schema for the top prediction with full details"""
    disease: str
    confidence: int = Field(..., ge=0, le=100, description="Confidence as percentage (0-100)")
    description: str
    recommendation: str


class PredictionResponseSchema(BaseModel):
    """Schema for the complete prediction response"""
    top_prediction: TopPredictionSchema
    predictions: List[PredictionItemSchema]
    ingredients_safe: List[str] = Field(default_factory=list)
    ingredients_avoid: List[str] = Field(default_factory=list)


class DiagnosisResultSchema(BaseModel):
    """
    Schema for diagnosis result from Vision API
    Returns the diagnosis with labels and confidence
    """
    diagnosis: str = Field(..., description="Detected skin condition")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for the diagnosis")
    labels: List[str] = Field(..., description="Labels detected by Vision API")
    prediction_id: Optional[str] = Field(None, description="Database ID of the stored prediction")


class PredictionResultSchema(BaseModel):
    """Schema for a single prediction result"""
    disease: str
    confidence: float = Field(..., ge=0, le=1)
    description: str
    recommendation: str
    ingredients: Optional[List[str]] = None


class IngredientsResponseSchema(BaseModel):
    """Schema for ingredient recommendations"""
    ingredients_safe: List[str] = Field(default_factory=list)
    ingredients_avoid: List[str] = Field(default_factory=list)


class DiagnosisPredictionSchema(BaseModel):
    """Standardized prediction response format for frontend"""
    top_prediction: PredictionResultSchema
    predictions: List[PredictionResultSchema]
    ingredients_safe: List[str] = Field(default_factory=list)
    ingredients_avoid: List[str] = Field(default_factory=list)


class DiagnosisResponseSchema(BaseModel):
    """Schema for immediate diagnosis response"""
    success: bool = True
    prediction: DiagnosisPredictionSchema
    message: Optional[str] = "Diagnosis completed successfully"


class ChatMessageSchema(BaseModel):
    """Schema for chatbot message"""
    message: str
    conversation_id: Optional[str] = None


class ChatResponseSchema(BaseModel):
    """Schema for chatbot response"""
    success: bool = True
    response: str
    conversation_id: Optional[str] = None
    suggestions: Optional[List[str]] = None


class HistoryItemSchema(BaseModel):
    """Schema for a single history item"""
    id: str
    prediction: str
    confidence: int = Field(..., ge=0, le=100, description="Confidence as percentage")
    model_version: str
    created_at: Optional[str] = None
    status: str


class HistoryResponseSchema(BaseModel):
    """Schema for prediction history response"""
    success: bool = True
    history: List[HistoryItemSchema]
    page: int
    page_size: int


class PredictionListResponseSchema(BaseModel):
    """Schema for prediction list response"""
    predictions: List[PredictionResultSchema]
    total: int
    page: int
    page_size: int
    has_more: bool


class PredictionResponseFullSchema(BaseModel):
    """Schema for prediction response"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., alias="_id")
    user_id: str
    image_url: str
    predictions: List[PredictionResultSchema]
    top_prediction: Optional[PredictionResultSchema] = None
    confidence_score: float
    body_part: Optional[str] = None
    skin_type: Optional[str] = None
    symptoms: Optional[List[str]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    model_version: str
    processing_time_ms: Optional[int] = None

