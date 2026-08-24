from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.metrics import prometheus_middleware
from src.api.router import router

app = FastAPI(title="personal chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(prometheus_middleware)

# Include router
app.include_router(router, prefix="/api/v1")


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def read_root():
    return {"message": "Welcome to the personal chatbot API!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}

