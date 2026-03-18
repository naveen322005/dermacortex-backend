"""
DermaCortex AI - Prediction Routes
Handle skin disease predictions and history
"""

from app.database import get_db
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    UploadFile,
    File
)

from typing import Optional
import base64
from datetime import datetime

from app.schemas.prediction import (
    DiagnosisRequestSchema,
    DiagnosisResponseSchema,
    PredictionListResponseSchema,
    PredictionResponseSchema,
    DiagnosisResultSchema,
    HistoryResponseSchema,
    PredictionResponseFullSchema
)

from app.services.prediction_service import (
    PredictionService,
    format_prediction_response,
    decimal_to_percentage
)
from app.services.vision_service import VisionService
from app.core.deps import get_current_user
from app.database import get_predictions_collection


# Router
router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


# =========================================================
# MOBILE APP ENDPOINT - Using Google Cloud Vision API
# =========================================================

@router.post("/diagnose", response_model=DiagnosisResultSchema)
async def diagnose(
    image: UploadFile = File(...),
    body_part: Optional[str] = Query(None, description="Body part affected"),
    skin_type: Optional[str] = Query(
        "normal",
        description="Skin type: normal, dry, oily, combination, sensitive"
    ),
    current_user: dict = Depends(get_current_user)
):
    """
    Submit a skin image for AI-powered disease diagnosis
    Uses Google Cloud Vision API for image analysis
    """

    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided"
        )

    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    content_type = image.content_type

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image type. Allowed types: JPEG, PNG, WebP"
        )

    try:
        image_bytes = await image.read()

        # Limit file size to 10MB
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image file too large. Maximum size is 10MB"
            )

        # Call Google Vision API
        vision_result, error = await VisionService.analyze_skin(image_bytes)

        if error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error
            )

        if not vision_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to analyze image"
            )

        # Convert image to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        predictions_collection = get_predictions_collection()

        prediction_doc = {
            "user_id": current_user["_id"],
            "image_url": f"data:{content_type};base64,{image_base64[:100]}...",
            "image_filename": image.filename,
            "predictions": [
                {
                    "disease": vision_result.diagnosis,
                    "confidence": vision_result.confidence,
                    "description": f"Analysis based on Vision API labels: {', '.join(vision_result.labels)}",
                    "recommendation": "Please consult a dermatologist for definitive diagnosis.",
                    "ingredients": []
                }
            ],
            "top_prediction": {
                "disease": vision_result.diagnosis,
                "confidence": vision_result.confidence,
                "description": f"Analysis based on Vision API labels: {', '.join(vision_result.labels)}",
                "recommendation": "Please consult a dermatologist for definitive diagnosis.",
                "ingredients": []
            },
            "confidence_score": vision_result.confidence,
            "vision_labels": vision_result.labels,
            "body_part": body_part,
            "skin_type": skin_type,
            "symptoms": [],
            "created_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
            "status": "completed",
            "error_message": None,
            "model_version": "2.0.0",
            "analysis_method": "google_vision_api",
            "processing_time_ms": 0
        }

        result = await predictions_collection.insert_one(prediction_doc)
        prediction_doc["_id"] = result.inserted_id

        return DiagnosisResultSchema(
            diagnosis=vision_result.diagnosis,
            confidence=vision_result.confidence,
            labels=vision_result.labels,
            prediction_id=str(result.inserted_id)
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process prediction: {str(e)}"
        )


# =========================================================
# SWAGGER TESTING ENDPOINT
# =========================================================

@router.post("/diagnose-upload")
async def diagnose_upload(
    image: UploadFile = File(...),
    body_part: Optional[str] = "unknown",
    skin_type: Optional[str] = "unknown",
    current_user: dict = Depends(get_current_user)
):
    """
    Submit a skin image for AI-powered disease diagnosis
    Returns Top 3 predictions with ingredient recommendations
    Confidence is returned as percentages (0-100)
    """

    try:
        image_bytes = await image.read()
        
        # Limit file size to 10MB
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image file too large. Maximum size is 10MB"
            )
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        prediction, error = await PredictionService.create_prediction(
            user_id=str(current_user["_id"]),
            image_base64=image_base64,
            body_part=body_part,
            skin_type=skin_type,
            symptoms=[]
        )

        if error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )

        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed"
            )

        # Use the service function to format the response
        # This converts confidence from decimals to percentages
        return format_prediction_response(prediction, top_n=3)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@router.get("/history", response_model=HistoryResponseSchema)
async def get_prediction_history(
    page: int = 1,
    page_size: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Get prediction history for the authenticated user
    Returns list of past predictions sorted by created_at descending
    """
    skip = (page - 1) * page_size

    # Use get_predictions_collection() to avoid db reference error
    predictions_collection = get_predictions_collection()

    # Sort by created_at descending (-1)
    predictions = await predictions_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1).skip(skip).limit(page_size).to_list(length=page_size)

    results = []

    for p in predictions:
        # Extract prediction data
        top_pred = p.get("top_prediction", {})
        
        # Convert confidence to percentage
        confidence_value = top_pred.get("confidence") if top_pred else p.get("confidence")
        confidence_percentage = decimal_to_percentage(confidence_value) if confidence_value else 0
        
        results.append({
            "id": str(p.get("_id")),
            "prediction": top_pred.get("disease") if top_pred else p.get("prediction", "Unknown"),
            "confidence": confidence_percentage,
            "model_version": p.get("model_version", "1.0.0"),
            "created_at": p.get("created_at").isoformat() if p.get("created_at") else None,
            "status": p.get("status", "completed")
        })

    return {
        "success": True,
        "history": results,
        "page": page,
        "page_size": page_size
    }


# =========================================================
# GET SINGLE PREDICTION
# =========================================================

@router.get("/{prediction_id}", response_model=PredictionResponseFullSchema)
async def get_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user)
):

    prediction = await PredictionService.get_prediction_by_id(
        prediction_id=prediction_id,
        user_id=current_user["_id"]
    )

    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )

    return prediction


# =========================================================
# HEALTH CHECK (PUBLIC ENDPOINT)
# =========================================================

@router.get("/health")
async def predictions_health_check():
    """
    Health check endpoint (no authentication required)
    """

    return {
        "status": "healthy",
        "service": "predictions",
        "model_version": "2.0.0",
        "analysis_method": "google_vision_api"
    }

