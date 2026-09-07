from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import datasets, explain, geocode, health, iceberg, optimize, route, seaice

app = FastAPI(
    title="Ocean Intelligence API",
    description="AI/ML-based Antarctic navigation decision-support backend.",
    version="0.1.0",
)

# Allow the Vite dev server (and common alternates) to call this API.
# Adjust/extend when the frontend is deployed somewhere else.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# Health check lives at the root (no /api prefix) so it's easy to hit directly.
app.include_router(health.router)

# Everything else lives under /api/...
app.include_router(geocode.router, prefix="/api")
app.include_router(route.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(seaice.router, prefix="/api")
app.include_router(iceberg.router, prefix="/api")
app.include_router(optimize.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
