"""NEXUS Railway Intelligence Platform — API Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NEXUS",
    description="AI-powered railway intelligence platform for predictive disruption management.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "NEXUS", "status": "operational", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
