"""
DermaCortex AI - Main Application
FastAPI backend for AI-Powered Skin Disease Detection & Dermatology Assistant
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.config import settings
from app.database import connect_to_mongodb, close_mongodb_connection
from app.routes import auth, predictions, chatbot


# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting DermaCortex AI Backend...")
    await connect_to_mongodb()
    yield
    # Shutdown
    print("🛑 Shutting down DermaCortex AI Backend...")
    await close_mongodb_connection()


# ==================== FastAPI App ====================

app = FastAPI(
    title="DermaCortex AI API",
    description="""
    ## AI-Powered Skin Disease Detection & Dermatology Assistant
    
    DermaCortex AI provides a comprehensive API for:
    
    - **JWT Authentication**: Secure user registration and login
    - **Skin Disease Prediction**: AI-powered diagnosis from skin images
    - **Confidence Scores**: Top 3 predictions with confidence levels
    - **Recommendations**: Personalized treatment recommendations
    - **Ingredient Analysis**: Active ingredient suggestions
    - **Dermatology Chatbot**: AI-powered dermatology Q&A
    - **Prediction History**: Track all past diagnoses
    
    ### Security
    - All endpoints (except health) require authentication
    - JWT tokens with access/refresh token support
    - Password hashing with bcrypt
    
    ### Limits
    - Max image size: 10MB
    - Rate limit: 100 requests/minute per user
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ==================== CORS Middleware ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request Logging ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    print(f"📝 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# ==================== Global Exception Handler ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions"""
    print(f"❌ Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


# ==================== Routes ====================

# Include routers
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(chatbot.router)


# ==================== Health Check ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "name": "DermaCortex AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "message": "Welcome to DermaCortex AI - AI-Powered Skin Disease Detection"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DermaCortex AI Backend",
        "version": "1.0.0",
        "timestamp": time.time()
    }


# ==================== Run Info ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

