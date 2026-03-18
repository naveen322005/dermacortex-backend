"""
DermaCortex AI - Vision Service
Google Cloud Vision API integration for skin analysis
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from google.cloud.vision_v1 import ImageAnnotatorClient
from google.cloud.vision_v1.types import Image

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VisionAnalysisResult:
    """Result from Vision API analysis"""
    labels: List[str]
    confidences: List[float]
    diagnosis: str
    confidence: float


# Mapping of Vision API labels to dermatology diagnoses
DERMATOLOGY_MAPPING = {
    # Acne-related labels
    "acne": {
        "diagnosis": "Acne Vulgaris",
        "description": "A common skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
        "confidence_boost": 0.2
    },
    "pimple": {
        "diagnosis": "Acne Vulgaris",
        "description": "A common skin condition with inflamed spots and blemishes.",
        "confidence_boost": 0.2
    },
    "blackhead": {
        "diagnosis": "Acne Vulgaris",
        "description": "Small bumps that appear on the skin due to clogged hair follicles.",
        "confidence_boost": 0.15
    },
    "whitehead": {
        "diagnosis": "Acne Vulgaris",
        "description": "Small bumps caused by clogged hair follicles.",
        "confidence_boost": 0.15
    },
    "sebum": {
        "diagnosis": "Acne Vulgaris",
        "description": "Oily skin condition that can lead to acne.",
        "confidence_boost": 0.1
    },
    
    # Eczema/Dermatitis labels
    "eczema": {
        "diagnosis": "Eczema (Atopic Dermatitis)",
        "description": "A condition that makes skin red, inflamed, and itchy.",
        "confidence_boost": 0.25
    },
    "dermatitis": {
        "diagnosis": "Contact Dermatitis",
        "description": "A red, itchy rash caused by direct contact with a substance.",
        "confidence_boost": 0.25
    },
    "rash": {
        "diagnosis": "Dermatitis",
        "description": "Red, inflamed skin condition with irritation.",
        "confidence_boost": 0.2
    },
    "redness": {
        "diagnosis": "Dermatitis",
        "description": "Skin inflammation characterized by redness.",
        "confidence_boost": 0.15
    },
    "inflammation": {
        "diagnosis": "Dermatitis",
        "description": "Inflamed skin condition.",
        "confidence_boost": 0.15
    },
    "itch": {
        "diagnosis": "Dermatitis",
        "description": "Itchy skin condition.",
        "confidence_boost": 0.1
    },
    "itchy": {
        "diagnosis": "Dermatitis",
        "description": "Itchy skin condition.",
        "confidence_boost": 0.1
    },
    
    # Psoriasis labels
    "psoriasis": {
        "diagnosis": "Psoriasis",
        "description": "A skin disease that causes red, itchy scaly patches.",
        "confidence_boost": 0.25
    },
    "scale": {
        "diagnosis": "Psoriasis",
        "description": "Scaly skin patches.",
        "confidence_boost": 0.15
    },
    "scaly": {
        "diagnosis": "Psoriasis",
        "description": "Rough, scaly skin patches.",
        "confidence_boost": 0.15
    },
    "plaque": {
        "diagnosis": "Psoriasis",
        "description": "Thickened skin patches.",
        "confidence_boost": 0.15
    },
    
    # Rosacea labels
    "rosacea": {
        "diagnosis": "Rosacea",
        "description": "A common skin condition causing redness and visible blood vessels.",
        "confidence_boost": 0.25
    },
    "facial redness": {
        "diagnosis": "Rosacea",
        "description": "Redness on the face.",
        "confidence_boost": 0.2
    },
    "flushing": {
        "diagnosis": "Rosacea",
        "description": "Persistent facial redness.",
        "confidence_boost": 0.2
    },
    
    # Melasma/Hyperpigmentation
    "melasma": {
        "diagnosis": "Melasma",
        "description": "Brown to gray-brown patches on the skin.",
        "confidence_boost": 0.25
    },
    "hyperpigmentation": {
        "diagnosis": "Melasma",
        "description": "Darkened skin patches.",
        "confidence_boost": 0.2
    },
    "pigmentation": {
        "diagnosis": "Melasma",
        "description": "Uneven skin tone.",
        "confidence_boost": 0.15
    },
    "dark spot": {
        "diagnosis": "Melasma",
        "description": "Darkened areas on the skin.",
        "confidence_boost": 0.15
    },
    "age spot": {
        "diagnosis": "Melasma",
        "description": "Dark spots related to sun exposure.",
        "confidence_boost": 0.15
    },
    
    # Vitiligo
    "vitiligo": {
        "diagnosis": "Vitiligo",
        "description": "A condition where skin loses its pigment cells.",
        "confidence_boost": 0.25
    },
    "depigmentation": {
        "diagnosis": "Vitiligo",
        "description": "Loss of skin color.",
        "confidence_boost": 0.2
    },
    "white patch": {
        "diagnosis": "Vitiligo",
        "description": "Light-colored skin patches.",
        "confidence_boost": 0.2
    },
    
    # Warts
    "wart": {
        "diagnosis": "Warts",
        "description": "Small, rough growths caused by HPV.",
        "confidence_boost": 0.25
    },
    "verruca": {
        "diagnosis": "Warts",
        "description": "Warts caused by human papillomavirus.",
        "confidence_boost": 0.25
    },
    
    # Herpes
    "herpes": {
        "diagnosis": "Herpes Simplex",
        "description": "Viral infection causing blisters or cold sores.",
        "confidence_boost": 0.25
    },
    "blister": {
        "diagnosis": "Herpes Simplex",
        "description": "Small fluid-filled skin lesions.",
        "confidence_boost": 0.2
    },
    "cold sore": {
        "diagnosis": "Herpes Simplex",
        "description": "Blisters around the mouth.",
        "confidence_boost": 0.25
    },
    
    # Hives
    "hives": {
        "diagnosis": "Hives (Urticaria)",
        "description": "Raised, itchy welts on the skin.",
        "confidence_boost": 0.25
    },
    "urticaria": {
        "diagnosis": "Hives (Urticaria)",
        "description": "Allergic skin reaction with itchy bumps.",
        "confidence_boost": 0.25
    },
    "welt": {
        "diagnosis": "Hives (Urticaria)",
        "description": "Raised itchy bumps on skin.",
        "confidence_boost": 0.2
    },
    
    # Skin cancer
    "carcinoma": {
        "diagnosis": "Skin Cancer Concern",
        "description": "Abnormal skin growth that may require medical attention.",
        "confidence_boost": 0.25
    },
    "melanoma": {
        "diagnosis": "Melanoma",
        "description": "Serious form of skin cancer from pigment cells.",
        "confidence_boost": 0.3
    },
    "tumor": {
        "diagnosis": "Skin Growth",
        "description": "Abnormal skin growth that should be evaluated.",
        "confidence_boost": 0.2
    },
    "lesion": {
        "diagnosis": "Skin Lesion",
        "description": "Abnormal skin area that should be examined.",
        "confidence_boost": 0.15
    },
    
    # Fungal infections
    "tinea": {
        "diagnosis": "Tinea (Ringworm)",
        "description": "Fungal infection causing ring-shaped rashes.",
        "confidence_boost": 0.25
    },
    "fungal": {
        "diagnosis": "Fungal Infection",
        "description": "Skin infection caused by fungi.",
        "confidence_boost": 0.2
    },
    "ringworm": {
        "diagnosis": "Tinea (Ringworm)",
        "description": "Circular fungal infection.",
        "confidence_boost": 0.25
    },
    "candidiasis": {
        "diagnosis": "Candida Infection",
        "description": "Yeast infection of the skin.",
        "confidence_boost": 0.25
    },
    
    # General skin conditions
    "dry skin": {
        "diagnosis": "Dry Skin (Xerosis)",
        "description": "Rough, flaky skin due to lack of moisture.",
        "confidence_boost": 0.15
    },
    "xerosis": {
        "diagnosis": "Dry Skin (Xerosis)",
        "description": "Abnormally dry skin.",
        "confidence_boost": 0.2
    },
    "oily skin": {
        "diagnosis": "Oily Skin",
        "description": "Skin with excess sebum production.",
        "confidence_boost": 0.15
    },
    "sensitive skin": {
        "diagnosis": "Sensitive Skin",
        "description": "Skin prone to irritation.",
        "confidence_boost": 0.15
    },
    "irritation": {
        "diagnosis": "Skin Irritation",
        "description": "Red, irritated skin.",
        "confidence_boost": 0.15
    },
    
    # Seborrheic conditions
    "seborrheic": {
        "diagnosis": "Seborrheic Dermatitis",
        "description": "Common skin condition affecting scalp and face.",
        "confidence_boost": 0.25
    },
    "dandruff": {
        "diagnosis": "Seborrheic Dermatitis",
        "description": "Flaky scalp condition.",
        "confidence_boost": 0.2
    },
    
    # General/healthy skin
    "skin": {
        "diagnosis": "Healthy Skin",
        "description": "Normal, healthy skin appearance.",
        "confidence_boost": 0.1
    },
    "face": {
        "diagnosis": "Healthy Skin",
        "description": "Facial skin appears normal.",
        "confidence_boost": 0.05
    },
    "human skin": {
        "diagnosis": "Healthy Skin",
        "description": "Normal skin tissue.",
        "confidence_boost": 0.05
    },
}


class VisionService:
    """Google Cloud Vision API service for skin analysis"""
    
    _client: Optional[ImageAnnotatorClient] = None
    
    @classmethod
    def get_client(cls) -> ImageAnnotatorClient:
        """
        Get or initialize the Vision API client
        Uses singleton pattern for efficiency
        """
        if cls._client is None:
            # Check for credentials in config
            if hasattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS') and settings.GOOGLE_APPLICATION_CREDENTIALS:
                cls._client = ImageAnnotatorClient()
            else:
                # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
                cls._client = ImageAnnotatorClient()
        
        return cls._client
    
    @classmethod
    async def analyze_skin(cls, image_bytes: bytes) -> Tuple[Optional[VisionAnalysisResult], Optional[str]]:
        """
        Analyze skin image using Google Cloud Vision API
        
        Args:
            image_bytes: Raw image bytes to analyze
            
        Returns:
            Tuple of (VisionAnalysisResult, error_message)
        """
        try:
            client = cls.get_client()
            
            # Create image object from bytes
            image = Image(content=image_bytes)
            
            # Perform label detection
            response = client.label_detection(image=image)
            labels = response.label_annotations
            
            if not labels:
                return None, "No labels detected in the image. Please upload a clearer image."
            
            # Extract labels and confidences
            detected_labels = []
            detected_confidences = []
            
            for label in labels:
                detected_labels.append(label.description.lower())
                detected_confidences.append(label.score)
            
            # Map to dermatology diagnosis
            diagnosis, confidence, matched_labels = cls._map_to_dermatology(
                detected_labels, 
                detected_confidences
            )
            
            result = VisionAnalysisResult(
                labels=matched_labels,
                confidences=[detected_confidences[detected_labels.index(l)] for l in matched_labels],
                diagnosis=diagnosis,
                confidence=confidence
            )
            
            return result, None
            
        except Exception as e:
            logger.error(f"Vision API error: {str(e)}")
            return None, f"Failed to analyze image: {str(e)}"
    
    @classmethod
    def _map_to_dermatology(
        cls, 
        labels: List[str], 
        confidences: List[float]
    ) -> Tuple[str, float, List[str]]:
        """
        Map Vision API labels to dermatology diagnoses
        
        Args:
            labels: List of detected label descriptions
            confidences: List of confidence scores for each label
            
        Returns:
            Tuple of (diagnosis_name, confidence_score, matched_labels)
        """
        # Find matching dermatology labels
        matched_conditions = []
        
        for i, label in enumerate(labels):
            label_lower = label.lower()
            
            # Check for direct matches
            for key, mapping in DERMATOLOGY_MAPPING.items():
                if key in label_lower or label_lower in key:
                    confidence_boost = mapping["confidence_boost"]
                    effective_confidence = min(confidences[i] + confidence_boost, 1.0)
                    
                    matched_conditions.append({
                        "diagnosis": mapping["diagnosis"],
                        "description": mapping["description"],
                        "confidence": effective_confidence,
                        "label": label
                    })
                    break
        
        if not matched_conditions:
            # No specific condition matched - default to healthy/skin check
            # Use the highest confidence skin-related label
            for i, label in enumerate(labels):
                label_lower = label.lower()
                if "skin" in label_lower or "face" in label_lower or "human" in label_lower:
                    return "Healthy Skin", confidences[i], [label]
            
            # Fallback: just return the top label
            if labels:
                return "Skin Analysis Required", confidences[0], [labels[0]]
            
            return "Unable to Analyze", 0.0, []
        
        # Sort by confidence score (descending)
        matched_conditions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Get the top match
        top_match = matched_conditions[0]
        
        # Collect all matched labels for the response
        matched_labels = [cond["label"] for cond in matched_conditions[:5]]  # Top 5 labels
        
        return top_match["diagnosis"], top_match["confidence"], matched_labels
    
    @classmethod
    def close_client(cls):
        """Close the Vision API client"""
        if cls._client is not None:
            # Google client doesn't have explicit close, just reset
            cls._client = None

