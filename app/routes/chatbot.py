"""
DermaCortex AI - Chatbot Routes
Handle dermatology-focused AI chatbot conversations
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from app.schemas.prediction import ChatMessageSchema, ChatResponseSchema
from app.services.chatbot_service import chatbot_service
from app.core.deps import get_current_user


# Router
router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


# ==================== New Simple Chatbot Endpoint ====================

class ChatbotRequest(BaseModel):
    """Request schema for POST /chatbot/ endpoint"""
    message: str = Field(..., min_length=1, description="User's question")
    context: Optional[str] = Field(None, description="Optional diagnosis result from skin analysis")


class ChatbotResponse(BaseModel):
    """Response schema for POST /chatbot/ endpoint"""
    reply: str = Field(..., description="AI generated response")


@router.post("/", response_model=ChatbotResponse)
async def chatbot(
    request: ChatbotRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Send a message to the DermaCortex AI chatbot
    
    Request:
        - message: User's dermatology question
        - context: Optional diagnosis result (e.g., "Acne (82% confidence)")
    
    Response:
        - reply: AI generated response
    """
    # Validate message is not empty (additional validation beyond Pydantic)
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    try:
        # Get response from chatbot service
        reply = await chatbot_service.chat_with_context(
            message=request.message,
            context=request.context
        )
        
        return ChatbotResponse(reply=reply)
        
    except Exception as e:
        print(f"Error in chatbot endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sorry, I couldn't process that request right now. Please try again."
        )


# ==================== Existing Legacy Endpoints ====================

class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None


@router.post("/chat", response_model=ChatResponseSchema)
async def chat(
    chat_request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Send a message to the dermatology AI chatbot
    Bot will only respond to dermatology-related queries
    """
    # Validate message
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    # Get response from chatbot
    response, suggestions = await chatbot_service.chat(
        user_message=chat_request.message,
        conversation_history=chat_request.conversation_history
    )
    
    return ChatResponseSchema(
        success=True,
        response=response,
        conversation_id=None,
        suggestions=suggestions
    )


@router.get("/suggestions", response_model=dict)
async def get_suggestions(
    current_user: dict = Depends(get_current_user)
):
    """
    Get suggested dermatology questions
    """
    suggestions = [
        {
            "category": "Common Conditions",
            "questions": [
                "What causes acne and how to treat it?",
                "How do I manage eczema flare-ups?",
                "What is the best treatment for psoriasis?"
            ]
        },
        {
            "category": "Skincare",
            "questions": [
                "What is the best skincare routine for dry skin?",
                "How to choose the right sunscreen?",
                "What ingredients should I look for in anti-aging products?"
            ]
        },
        {
            "category": "Hair & Nails",
            "questions": [
                "What causes hair loss?",
                "How to strengthen brittle nails?",
                "What are the best treatments for dandruff?"
            ]
        },
        {
            "category": "Prevention",
            "questions": [
                "How to check for skin cancer?",
                "What are the signs of melanoma?",
                "How to protect skin from sun damage?"
            ]
        }
    ]
    
    return {
        "success": True,
        "suggestions": suggestions
    }


@router.get("/health", response_model=dict)
async def chatbot_health_check():
    """
    Health check endpoint for chatbot service
    """
    return {
        "status": "healthy",
        "service": "chatbot",
        "type": "dermatology-focused AI"
    }

