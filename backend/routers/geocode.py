from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/geocode", tags=["geocode"])


@router.get("")
def geocode_place(place: str):
    """
    Will resolve a place name to coordinates.
    Implemented in Step 2 (route analyze endpoint).
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Geocoding not implemented yet.",
            "planned_step": "Step 2 - route analysis endpoint",
        },
    )
