from fastapi import APIRouter, HTTPException

from models.schemas import RouteAnalyzeRequest

router = APIRouter(prefix="/route", tags=["route"])


@router.get("/analyze")
def analyze_route(start: str, destination: str):
    """
    Will replace the mock routeData object currently fabricated inside
    RouteAnalysis.jsx. Response shape is already defined in
    models.schemas.RouteAnalyzeResponse so the frontend integration is a
    drop-in swap once this is implemented.
    Implemented in Step 2.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Route analysis not implemented yet.",
            "planned_step": "Step 2 - geocoding + distance/ETA",
        },
    )
