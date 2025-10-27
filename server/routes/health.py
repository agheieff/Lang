from __future__ import annotations

from typing import Dict

from fastapi import APIRouter


router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

