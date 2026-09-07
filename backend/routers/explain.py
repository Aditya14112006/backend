from fastapi import APIRouter, HTTPException

from models.schemas import ExplainRequest

router = APIRouter(prefix="/explain", tags=["explain"])


@router.post("")
def explain_prediction(request: ExplainRequest):
    """
    Will send prediction context to a local Ollama model and return a
    natural-language explanation of why the model produced that result.
    Implemented in Step 7 - depends on Steps 4-6 having real outputs to explain.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "detail": "Explanation endpoint not implemented yet.",
            "planned_step": "Step 7 - Ollama explanation",
        },
    )
