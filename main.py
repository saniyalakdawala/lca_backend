"""
Sustainability Intelligence Engine — FastAPI Backend
L&T Technology Services | Restricted Circulation [cite: 2, 6]
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# IMPORT FIX: Import only the router object from routes.py
from routes import router 

from config import settings
from startup import initialize_services

# Configure logging for POC status tracking [cite: 59]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Boot-time initialization: loads vector index for RAG and warms 
    SustainabilityScoreEngine[cite: 60, 64].
    """
    logger.info("🚀 Starting Sustainability Intelligence Engine...")
    await initialize_services()
    logger.info("✅ All services ready for LCA scoring.")
    yield
    logger.info("🛑 Shutting down.")

app = FastAPI(
    title="Sustainability Intelligence Engine",
    description="Automated eBOM → LCA scoring, RAG matching, and GHG hotspot detection [cite: 21, 120]",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

# CORS Configuration for your React + Vite frontend 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust to settings.ALLOWED_ORIGINS for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
# FIX: Use the single router that contains ebom, scoring, rag, and charts
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint to verify the API status."""
    return {
        "status": "online",
        "engine": "SustainabilityScoreEngine",
        "message": "Ready for TECHgium finale in Mysuru"
    }