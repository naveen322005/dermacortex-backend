"""
DermaCortex AI - Configuration Module
Handles environment variables and application settings
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MongoDB Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "dermacortex"
    
    # JWT Configuration
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    
    # Groq Configuration
    GROQ_API_KEY: Optional[str] = None
    
    # Google Cloud Vision Configuration
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Class configuration
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create singleton instance
settings = Settings()

