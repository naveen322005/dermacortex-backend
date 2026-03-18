from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

# Import routers
from app.routes import auth
from app.routes import chatbot
from app.routes import predictions
from app.database import connect_to_mongodb, close_mongodb_connection

app = FastAPI(
    title="DermaCortex AI",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    print("Connecting to MongoDB...")
    await connect_to_mongodb()


@app.on_event("shutdown")
async def shutdown():
    print("Closing MongoDB connection...")
    await close_mongodb_connection()
    
# Register routers
app.include_router(auth.router)
app.include_router(chatbot.router)
app.include_router(predictions.router)


@app.get("/")
def root():
    return {"message": "DermaCortex AI API running"}