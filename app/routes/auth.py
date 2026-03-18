"""
DermaCortex AI - Authentication Routes
Handle user registration, login, token refresh, and profile management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.user import (
    UserRegisterSchema,
    UserLoginSchema,
    RefreshTokenSchema,
    UserResponse,
    TokenResponse,
    MessageResponse,
    UserUpdateSchema
)
from app.services.auth_service import AuthService
from app.core.deps import get_current_user


# Router
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegisterSchema):
    """
    Register a new user account
    """
    # Register user
    user_response, error = await AuthService.register_user(user_data)
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Generate tokens
    tokens, _, error = await AuthService.login_user(
        UserLoginSchema(email=user_data.email, password=user_data.password)
    )
    
    if error or not tokens:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate tokens"
        )
    
    return tokens


@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLoginSchema):
    """
    Login with email and password
    Returns access and refresh tokens
    """
    tokens, user, error = await AuthService.login_user(login_data)
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error
        )
    
    if not tokens or not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
    
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_data: RefreshTokenSchema):
    """
    Refresh access token using refresh token
    """
    tokens, error = await AuthService.refresh_access_token(refresh_data.refresh_token)
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error
        )
    
    return tokens


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdateSchema,
    current_user: dict = Depends(get_current_user)
):
    """
    Update current user profile
    """
    # Filter out None values
    update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
    
    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid update data provided"
        )
    
    updated_user = await AuthService.update_user(current_user["_id"], update_dict)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )
    
    return updated_user


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout (client should discard tokens)
    """
    # In a production system, you might want to blacklist the token
    # For now, we just return a success message
    return MessageResponse(message="Successfully logged out")


@router.get("/health", response_model=MessageResponse)
async def auth_health_check():
    """
    Health check endpoint for authentication service
    """
    return MessageResponse(message="Authentication service is healthy")

