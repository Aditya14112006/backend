from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/predict", tags=["sea-ice"])


@router.post("/sea-ice")
def predict_sea_ice():
    """
    Will run a scikit-learn model trained on uploaded/sample ocean datasets
    to estimate sea-ice conditions along a route.
    Implemented in Step 4 - requires a dataset uploaded via Step 3 first.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Sea-ice prediction not implemented yet.",
            "planned_step": "Step 4 - sea-ice model",
        },
    )
