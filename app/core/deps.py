"""
DermaCortex AI - Dependencies Module
Authentication dependency injection
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from jose import JWTError, jwt

from app.config import settings
from app.database import get_users_collection
from app.core.security import verify_access_token

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get current authenticated user from JWT token
    """
    token = credentials.credentials
    
    # Verify token
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user email from token
    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Fetch user from database
    users_collection = get_users_collection()
    user = await users_collection.find_one({"email": email})
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Convert ObjectId to string
    user["_id"] = str(user["_id"])
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
):
    """
    Get current user if authenticated, otherwise return None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def validate_dermatology_query(query: str) -> bool:
    """
    Validate if the query is dermatology-related
    """
    # Dermatology-related keywords
    dermatology_keywords = [
        "skin", "dermat", "acne", "eczema", "psoriasis", "rash", "hives",
        "wart", "melanoma", "cancer", "mole", "scar", "dry skin", "oily skin",
        "aging", "wrinkle", "fine lines", "sun damage", " SPF", "sunblock",
        "sunscreen", "SPF", "UV", "ultraviolet", "dermatitis", "rosacea",
        "seborrheic", "keratosis", "lipoma", "cyst", "boil", "cellulitis",
        "impetigo", "herpes", "shingles", "chickenpox", "measles", "rubella",
        "lupus", "scleroderma", "vitiligo", "alopecia", "hair loss", "nail",
        "fungal", "infection", "bacterial", "viral", "parasitic", "allergy",
        "itch", "itchy", "redness", "inflammation", "swelling", "bump",
        "lesion", "patch", "spot", "discoloration", "pigment", "texture",
        "dryness", "oiliness", "sensitivity", "irritation", "burn", "cut",
        "abrasion", "wound", "healing", "treatment", "medication", "cream",
        "ointment", "gel", "lotion", "moisturizer", "cleanser", "serum",
        "retinoid", "retinol", "vitamin c", "aha", "bha", "niacinamide",
        "hydroquinone", "azelaic", "salicylic", "benzoyl peroxide", "clindamycin",
        "tretinoin", "adapalene", "differin", "minoxidil", "finasteride",
        "cosmetic", "procedure", "laser", "peel", "microneedling", "cryotherapy",
        "excision", "biopsy", "dermatologist", "dermatology", "consultation",
        "diagnosis", "prognosis", "symptom", "sign", "medical", "health",
        "body", "face", "neck", "arm", "leg", "hand", "foot", "scalp",
        "back", "chest", "stomach", "genital", "intimate", "sensitive area"
    ]
    
    query_lower = query.lower()
    
    # Check if any dermatology keyword is in the query
    for keyword in dermatology_keywords:
        if keyword in query_lower:
            return True
    
    return False

