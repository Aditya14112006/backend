from fastapi import APIRouter, HTTPException

from ml.iceberg_trajectory.model import get_service
from models.schemas import IcebergListResponse, TrajectoryRequest, TrajectoryResponse

router = APIRouter(prefix="/predict", tags=["iceberg"])


@router.get("/icebergs", response_model=IcebergListResponse)
def list_tracked_icebergs():
    """List every iceberg in the tracked dataset with its latest known position."""
    service = get_service()
    return IcebergListResponse(icebergs=service.list_icebergs())


@router.post("/iceberg-trajectory", response_model=TrajectoryResponse)
def predict_iceberg_trajectory(request: TrajectoryRequest):
    """
    Predict an iceberg's position `days_ahead` days from now.

    Returns the trained model's prediction AND two simple baselines
    (persistence, linear extrapolation) side by side, each tagged with its
    real validated error from training - because on this dataset the model
    does not clearly outperform the baselines. See `disclaimer` and
    `model_metadata` in the response.
    """
    service = get_service()

    if request.iceberg_id:
        try:
            return service.predict_for_iceberg(request.iceberg_id, request.days_ahead)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    required_manual = [
        request.manual_current_lat, request.manual_current_lon,
        request.manual_prev_lat, request.manual_prev_lon, request.manual_prev_days,
        request.manual_length_nm, request.manual_width_nm, request.manual_area_sqkm,
    ]
    if any(v is None for v in required_manual):
        raise HTTPException(
            status_code=422,
            detail="Either 'iceberg_id' or all manual_* fields must be supplied.",
        )

    try:
        return service.predict_manual(
            current_lat=request.manual_current_lat,
            current_lon=request.manual_current_lon,
            prev_lat=request.manual_prev_lat,
            prev_lon=request.manual_prev_lon,
            prev_days=request.manual_prev_days,
            length_nm=request.manual_length_nm,
            width_nm=request.manual_width_nm,
            area_sqkm=request.manual_area_sqkm,
            days_ahead=request.days_ahead,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
