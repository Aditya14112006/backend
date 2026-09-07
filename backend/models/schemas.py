"""
Pydantic schemas for the Ocean Intelligence backend.

These are defined up front (even for endpoints not implemented yet) so that:
  - the response shape for /api/route/analyze matches exactly what
    RouteAnalysis.jsx already expects on the frontend, and
  - every field that isn't backed by a real model/dataset yet is Optional
    and paired with a `note`, so we never invent a plausible-looking number.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    database: bool = Field(..., description="Whether SQLite was reachable")
    version: str


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    name: str
    latitude: float
    longitude: float


class TravelTime(BaseModel):
    hours: int
    minutes: int


class IcebergInfo(BaseModel):
    detected: Optional[int] = None
    confidence: Optional[float] = Field(
        None, description="Real model confidence (0-100). Never fabricated."
    )
    risk: Optional[str] = None
    note: Optional[str] = None


class OceanData(BaseModel):
    temperature: Optional[float] = None
    waveHeight: Optional[float] = None
    windSpeed: Optional[float] = None
    salinity: Optional[float] = None
    visibility: Optional[float] = None
    note: Optional[str] = None


class CycloneInfo(BaseModel):
    risk: Optional[str] = None
    message: Optional[str] = None


class SafeRouteInfo(BaseModel):
    risk: Optional[str] = None
    recommendation: Optional[str] = None
    status: Optional[str] = None
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# /api/route/analyze  (matches RouteAnalysis.jsx's routeData object 1:1)
# ---------------------------------------------------------------------------

class RouteAnalyzeRequest(BaseModel):
    start: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)


class RouteAnalyzeResponse(BaseModel):
    startLocation: Coordinates
    destinationLocation: Coordinates
    distance: float
    travelTime: TravelTime
    iceberg: IcebergInfo
    ocean: OceanData
    cyclone: CycloneInfo
    safeRoute: SafeRouteInfo


# ---------------------------------------------------------------------------
# /api/geocode
# ---------------------------------------------------------------------------

class GeocodeResponse(BaseModel):
    results: List[Coordinates]


# ---------------------------------------------------------------------------
# /api/datasets
# ---------------------------------------------------------------------------

class DatasetInfo(BaseModel):
    id: int
    filename: str
    dataset_type: str
    uploaded_at: str
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None


class DatasetUploadResponse(BaseModel):
    dataset: DatasetInfo
    message: str


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]


# ---------------------------------------------------------------------------
# /api/predict/iceberg-trajectory
# ---------------------------------------------------------------------------

class IcebergSummary(BaseModel):
    iceberg_id: str
    latest_date: str
    latitude: float
    longitude: float
    length_nm: float
    width_nm: float
    area_sqkm: float
    observation_count: int


class IcebergListResponse(BaseModel):
    icebergs: List[IcebergSummary]


class TrajectoryRequest(BaseModel):
    """
    Either supply iceberg_id (to predict from a tracked iceberg's own
    history) OR supply the manual_* fields (for an iceberg not in our
    dataset). If both are given, iceberg_id takes priority.
    """
    iceberg_id: Optional[str] = None
    days_ahead: int = Field(7, ge=1, le=60)

    manual_current_lat: Optional[float] = None
    manual_current_lon: Optional[float] = None
    manual_prev_lat: Optional[float] = None
    manual_prev_lon: Optional[float] = None
    manual_prev_days: Optional[int] = Field(None, gt=0)
    manual_length_nm: Optional[float] = None
    manual_width_nm: Optional[float] = None
    manual_area_sqkm: Optional[float] = None


class TrajectoryResponse(BaseModel):
    iceberg_id: Optional[str] = None
    last_observed_date: Optional[str] = None
    current_position: dict
    days_ahead: int
    predictions: dict
    recommended_method: str
    disclaimer: str
    model_metadata: dict


# ---------------------------------------------------------------------------
# /api/explain (Ollama)
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    context: str = Field(..., description="Structured summary of a prediction to explain")


class ExplainResponse(BaseModel):
    explanation: str
    model: str


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class NotImplementedResponse(BaseModel):
    detail: str
    planned_step: str
