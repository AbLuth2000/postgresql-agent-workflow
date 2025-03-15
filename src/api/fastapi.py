from fastapi import FastAPI
from src.api.routes import router  # Import routes

app = FastAPI(title="Agentic Workflow API", version="1.0")

# Register API routes
app.include_router(router)
