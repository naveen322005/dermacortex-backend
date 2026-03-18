"""
DermaCortex AI - User Schemas
Pydantic schemas for user-related requests and responses
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# ==================== Request Schemas ====================

class UserRegisterSchema(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)


class UserLoginSchema(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class RefreshTokenSchema(BaseModel):
    """Schema for token refresh"""
    refresh_token: str


class UserUpdateSchema(BaseModel):
    """Schema for user updates"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class ChangePasswordSchema(BaseModel):
    """Schema for password change"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


# ==================== Response Schemas ====================

class UserBaseResponse(BaseModel):
    """Base user response schema"""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True  # ⭐ IMPORTANT for MongoDB alias support
    )

    # MongoDB _id mapped to API id
    id: str = Field(..., alias="_id")

    email: EmailStr
    full_name: str
    avatar_url: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    is_active: bool
    is_verified: bool

    total_predictions: int = 0
    last_login: Optional[datetime] = None


class UserResponse(UserBaseResponse):
    """Full user response schema"""
    pass


class TokenResponse(BaseModel):
    """Token response schema"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class MessageResponse(BaseModel):
    """Generic message response"""

    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response schema"""

    detail: str
    success: bool = False
    error_code: Optional[str] = None