"""Employee Attrition Intelligence API.

Endpoints: /health, /predict, /explain, /simulate, /summary, /drivers/global, /employees, /employees/{id}.
Set API_KEY to require an X-API-Key header on everything except /health.
Set ALLOWED_ORIGINS (comma separated) to the frontend address(es) that may call the API.
"""

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from api.schemas import EmployeeInput, ExplainResponse, PredictResponse, SimulateRequest, SimulateResponse
from src.serving.demo_data import DemoStore
from src.serving.service import AttritionService, review_areas

logger = logging.getLogger("attrition_api")
state: dict = {}
API_KEY = os.getenv("API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = AttritionService.load()
        state["service"] = service
        try:
            state["store"] = DemoStore.load(service, list(EmployeeInput.model_fields))
        except FileNotFoundError:
            logger.warning("Demo data not found; run: python -m src.serving.make_demo_data")
            state["store"] = None
    except Exception:
        logger.exception("Could not load the model")
        state["service"] = None
        state["store"] = None
    yield
    state.clear()


app = FastAPI(
    title="Employee Attrition Intelligence API",
    version="0.2.0",
    description="Decision support only. Scores and explanations describe the model, not causes.",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_service() -> AttritionService:
    service = state.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service


def get_store() -> DemoStore:
    store = state.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail="Demo data not loaded")
    return store


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("service") is not None}


secured = APIRouter(dependencies=[Depends(require_key)])


@secured.post("/predict", response_model=PredictResponse)
def predict(employee: EmployeeInput):
    service = get_service()
    scored = service.score(pd.DataFrame([employee.model_dump()])).iloc[0]
    return PredictResponse(
        probability=float(scored["probability"]),
        risk_level=scored["risk_level"],
        flagged=bool(scored["flagged"]),
        cutoff=service.threshold,
    )


@secured.post("/explain", response_model=ExplainResponse)
def explain(employee: EmployeeInput):
    service = get_service()
    frame = pd.DataFrame([employee.model_dump()])
    scored = service.score(frame).iloc[0]
    drivers = service.drivers(frame, k=5)[0]
    return ExplainResponse(
        probability=float(scored["probability"]),
        risk_level=scored["risk_level"],
        drivers=drivers,
        review_areas=review_areas(drivers),
        note="These are the factors that matter most to the model's score for this employee. "
             "They describe the model, not causes, and are decision support, not decisions.",
    )


@secured.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest):
    service = get_service()
    base = request.employee.model_dump()
    try:
        EmployeeInput(**{**base, **request.changes})
    except ValidationError as exc:
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        raise HTTPException(status_code=422, detail=detail)
    return service.simulate(base, request.changes)


@secured.get("/summary")
def summary():
    return get_store().summary()


@secured.get("/drivers/global")
def global_drivers():
    return get_store().drivers()


@secured.get("/employees")
def list_employees(
    risk_level: str | None = Query(default=None, pattern="^(Low|Medium|High)$"),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return get_store().list_employees(risk_level, limit, offset)


@secured.get("/employees/{employee_number}")
def get_employee(employee_number: int):
    result = get_store().employee(employee_number)
    if result is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return result


app.include_router(secured)
