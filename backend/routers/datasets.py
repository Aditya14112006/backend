from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload")
def upload_dataset(file: UploadFile):
    """
    Will accept a CSV, store it under backend/data/uploads, register it in
    SQLite, and return row/column summary via Pandas.
    Implemented in Step 3.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Dataset upload not implemented yet.",
            "planned_step": "Step 3 - CSV upload",
        },
    )


@router.get("")
def list_datasets():
    """
    Will list previously uploaded datasets from SQLite.
    Implemented in Step 3.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Dataset listing not implemented yet.",
            "planned_step": "Step 3 - CSV upload",
        },
    )
