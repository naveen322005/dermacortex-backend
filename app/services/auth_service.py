"""
DermaCortex AI - Authentication Service
Handles user authentication and authorization logic
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import get_users_collection
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    verify_refresh_token
)
from app.config import settings
from app.schemas.user import (
    UserRegisterSchema,
    UserLoginSchema,
    TokenResponse,
    UserResponse
)


class AuthService:
    """Authentication service class"""
    
    @staticmethod
    async def register_user(user_data: UserRegisterSchema) -> Tuple[Optional[UserResponse], Optional[str]]:
        """
        Register a new user
        Returns: (user_response, error_message)
        """
        users_collection = get_users_collection()
        
        # Check if user already exists
        existing_user = await users_collection.find_one({"email": user_data.email})
        if existing_user:
            return None, "Email already registered"
        
        # Hash password
        password_hash = get_password_hash(user_data.password)
        
        # Create user document
        user_doc = {
            "email": user_data.email,
            "password_hash": password_hash,
            "full_name": user_data.full_name,
            "avatar_url": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True,
            "is_verified": False,
            "preferences": {},
            "total_predictions": 0,
            "last_login": None
        }
        
        # Insert user
        result = await users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        
        # Convert to response
        user_response = UserResponse(
            _id=str(user_doc["_id"]),
            email=user_doc["email"],
            full_name=user_doc["full_name"],
            avatar_url=user_doc["avatar_url"],
            created_at=user_doc["created_at"],
            updated_at=user_doc["updated_at"],
            is_active=user_doc["is_active"],
            is_verified=user_doc["is_verified"],
            total_predictions=user_doc["total_predictions"],
            last_login=user_doc["last_login"]
        )
        
        return user_response, None
    
    @staticmethod
    async def login_user(login_data: UserLoginSchema) -> Tuple[Optional[TokenResponse], Optional[UserResponse], Optional[str]]:
        """
        Authenticate user and generate tokens
        Returns: (tokens, user_response, error_message)
        """
        users_collection = get_users_collection()
        
        # Find user by email
        user = await users_collection.find_one({"email": login_data.email})
        if not user:
            return None, None, "Invalid email or password"
        
        # Verify password
        if not verify_password(login_data.password, user["password_hash"]):
            return None, None, "Invalid email or password"
        
        # Check if user is active
        if not user.get("is_active", True):
            return None, None, "Account is deactivated"
        
        # Update last login
        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Generate tokens
        access_token = create_access_token(data={"sub": user["email"]})
        refresh_token = create_refresh_token(data={"sub": user["email"]})
        
        token_response = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
        # Create user response
        user_response = UserResponse(
            id=str(user["_id"]),
            email=user["email"],
            full_name=user["full_name"],
            avatar_url=user.get("avatar_url"),
            created_at=user["created_at"],
            updated_at=user["updated_at"],
            is_active=user["is_active"],
            is_verified=user["is_verified"],
            total_predictions=user.get("total_predictions", 0),
            last_login=user.get("last_login")
        )
        
        return token_response, user_response, None
    
    @staticmethod
    async def refresh_access_token(refresh_token: str) -> Tuple[Optional[TokenResponse], Optional[str]]:
        """
        Refresh access token using refresh token
        Returns: (tokens, error_message)
        """
        # Verify refresh token
        payload = verify_refresh_token(refresh_token)
        if not payload:
            return None, "Invalid or expired refresh token"
        
        # Get user email
        email = payload.get("sub")
        if not email:
            return None, "Invalid token payload"
        
        # Verify user exists
        users_collection = get_users_collection()
        user = await users_collection.find_one({"email": email})
        if not user:
            return None, "User not found"
        
        # Generate new tokens
        access_token = create_access_token(data={"sub": email})
        new_refresh_token = create_refresh_token(data={"sub": email})
        
        token_response = TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
        return token_response, None
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[dict]:
        """Get user by email"""
        users_collection = get_users_collection()
        user = await users_collection.find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])
        return user
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[dict]:
        """Get user by ID"""
        users_collection = get_users_collection()
        try:
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception:
            return None
    
    @staticmethod
    async def update_user(user_id: str, update_data: dict) -> Optional[dict]:
        """Update user data"""
        users_collection = get_users_collection()
        update_data["updated_at"] = datetime.utcnow()
        
        result = await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return await AuthService.get_user_by_id(user_id)
        return None
    
    @staticmethod
    async def increment_prediction_count(user_id: str) -> bool:
        """Increment user's prediction count"""
        users_collection = get_users_collection()
        result = await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$inc": {"total_predictions": 1}}
        )
        return result.modified_count > 0

