from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/route", tags=["optimize"])


@router.post("/optimize")
def optimize_route():
    """
    Will compute a safe/fuel-efficient route corridor over a hazard-cost
    grid built from the sea-ice and iceberg-trajectory model outputs.
    Implemented in Step 6 - depends on Steps 4 and 5.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Route optimization not implemented yet.",
            "planned_step": "Step 6 - route optimizer",
        },
    )
