"""
DermaCortex AI - Database Module
MongoDB async connection using Motor
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from fastapi import Depends
from app.config import settings

# MongoDB Client
client: Optional[AsyncIOMotorClient] = None
database: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongodb():
    """Establish MongoDB connection"""
    global client, database

    client = AsyncIOMotorClient(settings.MONGODB_URL)
    database = client[settings.MONGODB_DB_NAME]

    # Create indexes for better query performance
    await database.users.create_index("email", unique=True)
    await database.predictions.create_index("user_id")
    await database.predictions.create_index("created_at")

    print(f"✅ Connected to MongoDB: {settings.MONGODB_DB_NAME}")


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global client

    if client:
        client.close()
        print("✅ MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    if database is None:
        raise RuntimeError(
            "Database not initialized. Call connect_to_mongodb first."
        )
    return database


# ✅ FastAPI dependency (THIS FIXES YOUR ERROR)
async def get_db():
    """FastAPI dependency to get DB instance"""
    return get_database()


# Collection helpers
def get_users_collection():
    """Get users collection"""
    return get_database().users


def get_predictions_collection():
    """Get predictions collection"""
    return get_database().predictions