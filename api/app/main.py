"""
API de cobertura de inventario — M5 / Walmart.

Flujo: la página web (o cualquier cliente) manda tienda, producto e inventario →
la API busca las features "al día de hoy" del producto-tienda (gold_latest) →
llama al endpoint publicado en Databricks Model Serving → calcula cobertura y pedido sugerido.

Documentación interactiva (Swagger/OpenAPI): <url-de-la-app>/docs
"""
import io
import logging
import math
from contextlib import asynccontextmanager
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from databricks.sdk.core import Config

log = logging.getLogger("m5-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE = Path(__file__).parent
HORIZON = 28
FEATS = ["item_id_idx", "dept_id_idx", "cat_id_idx", "store_id_idx", "state_id_idx",
         "m7", "m14", "m28", "m56", "m112", "std28", "zf28", "zf112", "days_since_sale", "ly28",
         "price", "price_norm", "price_chg28", "snap_days", "event_days", "month"]
Z = {"90": 1.2816, "95": 1.6449, "98": 2.0537, "99": 2.3263}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    serving_endpoint: str = "m5-demand"
    latest_volume_path: str = "/Volumes/workspace/m5/raw/exports/gold_latest.csv"
    local_latest_file: str = str(BASE.parent / "data" / "gold_latest.csv.gz")
    endpoint_timeout: float = 60.0


settings = Settings()
STATE: dict = {"latest": None, "source": None}
_CFG: dict = {}


# ---------------------------------------------------------------- autenticación con Databricks
def db_config() -> Config | None:
    """Credenciales vía databricks-sdk.
    - En Databricks Apps: identidad de servicio de la app (DATABRICKS_CLIENT_ID/SECRET, inyectadas solas).
    - En local/Docker: DATABRICKS_HOST + DATABRICKS_TOKEN en .env."""
    if "cfg" not in _CFG:
        try:
            _CFG["cfg"] = Config()
            _CFG["cfg"].authenticate()
        except Exception as e:  # noqa: BLE001
            log.warning("Databricks no configurado: %s", e)
            _CFG["cfg"] = None
    return _CFG["cfg"]


def db_request_parts() -> tuple[str, dict]:
    cfg = db_config()
    if cfg is None:
        raise HTTPException(503, "La API no tiene credenciales de Databricks configuradas.")
    return cfg.host.rstrip("/"), cfg.authenticate()


# ---------------------------------------------------------------- carga de features
def load_latest() -> tuple[pd.DataFrame, str]:
    """1) Archivo empaquetado con la app (data/gold_latest.csv.gz, lo escribe el notebook 06).
       2) Si no existe, lo descarga del Volume con la Files API."""
    local = Path(settings.local_latest_file)
    if local.exists():
        return pd.read_csv(local), "archivo_app"
    if db_config() is not None:
        host, headers = db_request_parts()
        r = httpx.get(f"{host}/api/2.0/fs/files{settings.latest_volume_path}", headers=headers, timeout=60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content)), "databricks_volume"
    raise RuntimeError("No hay gold_latest: corre el notebook 06 o configura credenciales de Databricks")


def refresh_latest() -> bool:
    try:
        df, src = load_latest()
        df["origin"] = pd.to_datetime(df["origin"]).dt.date
        STATE["latest"] = df.set_index(["store_id", "item_id"]).sort_index()
        STATE["source"] = src
        log.info("gold_latest: %s filas desde %s", len(df), src)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Sin features: %s", e)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh_latest()          # si falla, se reintenta en la primera petición
    yield


app = FastAPI(
    title="M5 · Cobertura de inventario",
    version="1.0.0",
    description="Pronóstico de demanda a 28 días (LightGBM servido en Databricks Model Serving) "
                "y cálculo de días de cobertura y pedido sugerido por producto-tienda.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


# ---------------------------------------------------------------- esquemas
class ServiceLevel(str, Enum):
    s90 = "90"
    s95 = "95"
    s98 = "98"
    s99 = "99"


class ForecastRequest(BaseModel):
    store_id: str = Field(..., pattern=r"^(CA|TX|WI)_[1-4]$", examples=["CA_3"],
                          description="Tienda, ej. CA_3")
    item_id: str = Field(..., pattern=r"^(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}$", examples=["FOODS_3_090"],
                         description="Producto, ej. FOODS_3_090")
    inventory_units: float = Field(..., ge=0, le=1_000_000, description="Inventario disponible hoy (unidades)")
    lead_time_days: int = Field(7, ge=0, le=60, description="Días que tarda en llegar un pedido")
    coverage_target_days: int = Field(14, ge=1, le=56, description="Días de venta que debe cubrir el pedido")
    service_level: ServiceLevel = Field(ServiceLevel.s95, description="Nivel de servicio para stock de seguridad (%)")


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    as_of: date = Field(description="Fecha 'hoy' del dataset (día siguiente al último dato)")
    forecast_28d_units: float = Field(description="Demanda pronosticada para los próximos 28 días")
    forecast_daily_units: float
    last_28d_units: float = Field(description="Unidades vendidas en los últimos 28 días (referencia)")
    trend_pct: float = Field(description="Cambio del pronóstico vs. últimos 28 días")
    coverage_days: float | None = Field(description="Días que alcanza el inventario actual (None = sin demanda)")
    stockout_date: date | None
    risk: Literal["alto", "medio", "bajo", "sin_demanda"]
    safety_stock_units: float
    reorder_point_units: float
    suggested_order_units: int
    price: float | None
    model_endpoint: str


# ---------------------------------------------------------------- endpoint de Databricks
async def predict(features: dict) -> float:
    """Llama al endpoint publicado. Reintenta 1 vez ante cold start (429/503/504/timeout)."""
    host, headers = db_request_parts()
    url = f"{host}/serving-endpoints/{settings.serving_endpoint}/invocations"
    payload = {"dataframe_records": [features]}
    last_err = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=settings.endpoint_timeout) as client:
                r = await client.post(url, json=payload, headers=headers)
            if r.status_code in (429, 503, 504) and attempt == 1:
                last_err = f"HTTP {r.status_code}"
                log.warning("Endpoint respondió %s (posible cold start); reintentando…", r.status_code)
                continue
            if r.status_code == 401 or r.status_code == 403:
                raise HTTPException(502, "Databricks rechazó las credenciales (en Apps: la app necesita permiso 'Can query' "
                                         "sobre el endpoint; en local: revisa DATABRICKS_TOKEN).")
            if r.status_code == 404:
                raise HTTPException(502, f"No existe el endpoint '{settings.serving_endpoint}' o está detenido.")
            r.raise_for_status()
            data = r.json()
            preds = data.get("predictions") if isinstance(data, dict) else None
            if not isinstance(preds, list) or len(preds) != 1:
                raise HTTPException(502, f"Respuesta inválida del modelo: {str(data)[:200]}")
            value = preds[0]
            if isinstance(value, list):
                value = value[0]
            value = float(value)
            if math.isnan(value) or math.isinf(value) or value < 0:
                raise HTTPException(502, f"El modelo devolvió un valor inválido: {value}")
            return value
        except httpx.TimeoutException:
            last_err = "timeout"
            log.warning("Timeout al llamar al endpoint (intento %s)", attempt)
        except httpx.RequestError as e:
            raise HTTPException(502, f"No se pudo conectar con Databricks: {e.__class__.__name__}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"El endpoint devolvió error HTTP {e.response.status_code}.")
    raise HTTPException(504, f"El modelo no respondió a tiempo ({last_err}). Si estaba en reposo "
                             "(scale to zero), espera 1–2 minutos y vuelve a intentar.")


# ---------------------------------------------------------------- rutas
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health", tags=["sistema"])
def health():
    df = STATE["latest"]
    return {"status": "ok" if df is not None else "sin_features",
            "features_source": STATE["source"], "rows": 0 if df is None else len(df),
            "endpoint": settings.serving_endpoint,
            "databricks_configured": db_config() is not None}


def _latest() -> pd.DataFrame:
    if STATE["latest"] is None and not refresh_latest():
        raise HTTPException(503, "La API no cargó las features (gold_latest). Revisa /health.")
    return STATE["latest"]


@app.get("/stores", tags=["catálogo"], response_model=list[str])
def stores():
    return sorted(_latest().index.get_level_values("store_id").unique().tolist())


@app.get("/items", tags=["catálogo"], response_model=list[str])
def items(store_id: str = Query(..., pattern=r"^(CA|TX|WI)_[1-4]$"),
          q: str = Query("", max_length=30, description="Filtro por texto, ej. FOODS_3"),
          limit: int = Query(50, ge=1, le=500)):
    df = _latest()
    if store_id not in df.index.get_level_values("store_id"):
        raise HTTPException(404, f"Tienda {store_id} no encontrada")
    ids = df.loc[store_id].index
    ids = ids[ids.str.contains(q.upper(), regex=False)] if q else ids
    return ids[:limit].tolist()


@app.post("/forecast", tags=["pronóstico"], response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    df = _latest()
    try:
        row = df.loc[(req.store_id, req.item_id)]
    except KeyError:
        raise HTTPException(404, f"No hay datos para {req.item_id} en {req.store_id}")

    features = {k: (None if pd.isna(row[k]) else float(row[k])) for k in FEATS}
    f28 = await predict(features)

    daily = f28 / HORIZON
    last28 = float(row["m28"] * HORIZON) if not pd.isna(row["m28"]) else 0.0
    as_of = row["origin"]

    # Stock de seguridad: z · σ_diaria · √(lead time + días a cubrir)
    sigma = 0.0 if pd.isna(row["std28"]) else float(row["std28"])
    horizon_days = req.lead_time_days + req.coverage_target_days
    safety = Z[req.service_level.value] * sigma * math.sqrt(max(horizon_days, 1))
    reorder_point = daily * req.lead_time_days + safety

    if daily <= 1e-6:
        coverage, stockout, risk = None, None, "sin_demanda"
    else:
        coverage = req.inventory_units / daily
        stockout = as_of + timedelta(days=int(coverage)) if coverage < 3650 else None
        # alto: se agota antes de que llegue un pedido · medio: ya cruzó el punto de reorden
        risk = "alto" if coverage < req.lead_time_days else ("medio" if req.inventory_units <= reorder_point else "bajo")

    # Política de punto de reorden: solo se pide al cruzar el punto de reorden;
    # la cantidad lleva el inventario a cubrir lead time + días objetivo + seguridad.
    order = 0.0 if risk in ("bajo", "sin_demanda") else max(0.0, daily * horizon_days + safety - req.inventory_units)

    return ForecastResponse(
        store_id=req.store_id, item_id=req.item_id, as_of=as_of,
        forecast_28d_units=round(f28, 1), forecast_daily_units=round(daily, 2),
        last_28d_units=round(last28, 1),
        trend_pct=round((f28 / last28 - 1) * 100, 1) if last28 > 0 else 0.0,
        coverage_days=None if coverage is None else round(coverage, 1),
        stockout_date=stockout, risk=risk,
        safety_stock_units=round(safety, 1), reorder_point_units=round(reorder_point, 1),
        suggested_order_units=int(math.ceil(order)),
        price=None if pd.isna(row["price"]) else float(row["price"]),
        model_endpoint=settings.serving_endpoint,
    )
