"""
DermaCortex AI - User Model
MongoDB document schema for users
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User document model"""
    
    email: EmailStr = Field(..., unique=True)
    password_hash: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    avatar_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    
    # User preferences
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Statistics
    total_predictions: int = Field(default=0)
    last_login: Optional[datetime] = None
    
    class Config:
        """Pydantic config"""
        from_attributes = True


class UserInDB(User):
    """User model as stored in database"""
    password_hash: str


class UserResponse(BaseModel):
    """User response model (without sensitive data)"""
    
    id: str = Field(..., alias="_id")
    email: EmailStr
    full_name: str
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    is_verified: bool
    total_predictions: int
    last_login: Optional[datetime] = None
    
    class Config:
        """Pydantic config"""
        populate_by_name = True
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Register request model"""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str


class UserUpdate(BaseModel):
    """User update request model"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

