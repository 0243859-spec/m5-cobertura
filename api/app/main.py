"""
API de cobertura de inventario — M5 / Walmart.

Flujo: la página web (o cualquier cliente) manda tienda, producto e inventario →
la API busca las features "al día de hoy" del producto-tienda (gold_latest) →
llama al endpoint publicado en Databricks Model Serving → calcula cobertura y pedido sugerido.

Además:
- /sales            ventas históricas (gold_sales_daily) con filtros, para explorar los datos.
- /store-forecast   pronóstico de TODOS los productos de una tienda (una llamada por lote al endpoint).

Documentación interactiva (Swagger/OpenAPI): <url-de-la-app>/docs
"""
import io
import logging
import math
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
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
BATCH = 1000                 # filas por llamada al endpoint en el pronóstico por tienda
CACHE_TTL = 6 * 3600         # segundos que se guarda el pronóstico de una tienda


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    serving_endpoint: str = "m5-demand"
    volume_dir: str = "/Volumes/workspace/m5/raw/exports"
    local_data_dir: str = str(BASE.parent / "data")
    endpoint_timeout: float = 60.0


settings = Settings()
STATE: dict = {"latest": None, "sales": None, "source": {}}
CACHE: dict = {}
_CFG: dict = {}


# ---------------------------------------------------------------- autenticación con Databricks
def db_config() -> Config | None:
    """Credenciales vía databricks-sdk.
    - En local/Docker: DATABRICKS_HOST + DATABRICKS_TOKEN en .env.
    - En Databricks Apps: identidad de servicio de la app (inyectada automáticamente)."""
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


# ---------------------------------------------------------------- carga de datos
def load_table(name: str) -> tuple[pd.DataFrame, str]:
    """1) Copia local en data/<name>.csv(.gz).  2) Si no existe, la descarga del Volume (Files API)."""
    for ext in (".csv.gz", ".csv"):
        local = Path(settings.local_data_dir) / f"{name}{ext}"
        if local.exists():
            return pd.read_csv(local), "archivo_local"
    if db_config() is not None:
        host, headers = db_request_parts()
        r = httpx.get(f"{host}/api/2.0/fs/files{settings.volume_dir}/{name}.csv", headers=headers, timeout=60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content)), "databricks_volume"
    raise RuntimeError(f"No hay {name}: corre el notebook 06 o configura credenciales de Databricks")


def refresh_latest() -> bool:
    try:
        df, src = load_table("gold_latest")
        df["origin"] = pd.to_datetime(df["origin"]).dt.date
        STATE["latest"] = df.set_index(["store_id", "item_id"]).sort_index()
        STATE["source"]["gold_latest"] = src
        log.info("gold_latest: %s filas desde %s", len(df), src)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Sin features: %s", e)
        return False


def refresh_sales() -> bool:
    try:
        df, src = load_table("gold_sales_daily")
        df["date"] = pd.to_datetime(df["date"])
        STATE["sales"] = df
        STATE["source"]["gold_sales_daily"] = src
        log.info("gold_sales_daily: %s filas desde %s", len(df), src)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Sin ventas históricas: %s", e)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh_latest()          # si falla, se reintenta en la primera petición
    refresh_sales()
    yield


app = FastAPI(
    title="M5 · Cobertura de inventario",
    version="1.1.0",
    description="Pronóstico de demanda a 28 días (LightGBM servido en Databricks Model Serving), "
                "cálculo de cobertura y pedido sugerido por producto-tienda, exploración de ventas "
                "históricas y pronóstico por tienda.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

# ---------------------------------------------------------------- mensajes de validación en español
_ERR_ES = {
    "missing": "es obligatorio",
    "string_pattern_mismatch": "tiene un formato inválido",
    "greater_than_equal": "debe ser mayor o igual a {ge}",
    "less_than_equal": "debe ser menor o igual a {le}",
    "int_parsing": "debe ser un número entero", "int_from_float": "debe ser un número entero",
    "float_parsing": "debe ser un número", "enum": "no es un valor permitido",
    "literal_error": "no es un valor permitido", "date_from_datetime_parsing": "debe ser una fecha AAAA-MM-DD",
    "date_parsing": "debe ser una fecha AAAA-MM-DD", "json_invalid": "el cuerpo no es un JSON válido",
}


@app.exception_handler(RequestValidationError)
async def validation_es(request: Request, exc: RequestValidationError):
    detalles = []
    for e in exc.errors():
        campo = str(e["loc"][-1]) if e.get("loc") else "entrada"
        msg = _ERR_ES.get(e.get("type"), "es inválido")
        try:
            msg = msg.format(**(e.get("ctx") or {}))
        except (KeyError, IndexError):
            pass
        detalles.append({"loc": list(e.get("loc", [])), "msg": msg, "type": e.get("type")})
    return JSONResponse(status_code=422, content={"detail": detalles})



# ---------------------------------------------------------------- esquemas
STORE_RE = r"^(CA|TX|WI)_[1-4]$"
CAT_RE = r"^(FOODS|HOBBIES|HOUSEHOLD)$"


class NivelServicio(str, Enum):
    """Nivel de servicio (%)"""
    s90 = "90"
    s95 = "95"
    s98 = "98"
    s99 = "99"


class SolicitudCobertura(BaseModel):
    store_id: str = Field(..., pattern=STORE_RE, examples=["CA_3"], description="Tienda, ej. CA_3")
    item_id: str = Field(..., pattern=r"^(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}$", examples=["FOODS_3_090"],
                         description="Producto, ej. FOODS_3_090")
    inventory_units: float = Field(..., ge=0, le=1_000_000, description="Inventario disponible hoy (unidades)")
    lead_time_days: int = Field(7, ge=0, le=60, description="Días que tarda en llegar un pedido")
    coverage_target_days: int = Field(14, ge=1, le=56, description="Días de venta que debe cubrir el pedido")
    service_level: NivelServicio = Field(NivelServicio.s95, description="Nivel de servicio para stock de seguridad (%)")


class RespuestaCobertura(BaseModel):
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


class PuntoVentas(BaseModel):
    period: date
    series: str
    value: float


class RespuestaVentas(BaseModel):
    metric: Literal["units", "revenue"]
    grain: Literal["day", "week", "month"]
    group_by: Literal["total", "state", "store", "category"]
    date_from: date
    date_to: date
    total: float
    avg_per_day: float
    points: list[PuntoVentas]


class PronosticoProducto(BaseModel):
    item_id: str
    category: str
    dept: str
    last_28d_units: float
    forecast_28d_units: float
    change_units: float
    change_pct: float | None
    price: float | None


class PronosticoGrupo(BaseModel):
    name: str
    last_28d_units: float
    forecast_28d_units: float
    change_pct: float | None


class RespuestaPronosticoTienda(BaseModel):
    store_id: str
    as_of: date
    n_items: int
    last_28d_units: float
    forecast_28d_units: float
    change_pct: float | None
    by_category: list[PronosticoGrupo]
    by_dept: list[PronosticoGrupo]
    items: list[PronosticoProducto]
    cached: bool
    model_endpoint: str


# ---------------------------------------------------------------- endpoint de Databricks
async def predict_many(records: list[dict]) -> list[float]:
    """Llama al endpoint publicado con 1..N filas. Reintenta 1 vez ante cold start (429/503/504/timeout)."""
    host, headers = db_request_parts()
    url = f"{host}/serving-endpoints/{settings.serving_endpoint}/invocations"
    payload = {"dataframe_records": records}
    last_err = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=settings.endpoint_timeout) as client:
                r = await client.post(url, json=payload, headers=headers)
            if r.status_code in (429, 503, 504) and attempt == 1:
                last_err = f"HTTP {r.status_code}"
                log.warning("Endpoint respondió %s (posible cold start); reintentando…", r.status_code)
                continue
            if r.status_code in (401, 403):
                raise HTTPException(502, "Databricks rechazó las credenciales (revisa DATABRICKS_TOKEN y su vigencia).")
            if r.status_code == 404:
                raise HTTPException(502, f"No existe el endpoint '{settings.serving_endpoint}' o está detenido.")
            r.raise_for_status()
            data = r.json()
            preds = data.get("predictions") if isinstance(data, dict) else None
            if not isinstance(preds, list) or len(preds) != len(records):
                raise HTTPException(502, f"Respuesta inválida del modelo: {str(data)[:200]}")
            out = []
            for v in preds:
                v = float(v[0] if isinstance(v, list) else v)
                if math.isnan(v) or math.isinf(v) or v < 0:
                    raise HTTPException(502, f"El modelo devolvió un valor inválido: {v}")
                out.append(v)
            return out
        except httpx.TimeoutException:
            last_err = "timeout"
            log.warning("Timeout al llamar al endpoint (intento %s)", attempt)
        except httpx.RequestError as e:
            raise HTTPException(502, f"No se pudo conectar con Databricks: {e.__class__.__name__}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"El endpoint devolvió error HTTP {e.response.status_code}.")
    raise HTTPException(504, f"El modelo no respondió a tiempo ({last_err}). Si estaba en reposo "
                             "(scale to zero), espera 1–2 minutos y vuelve a intentar.")


async def predict(features: dict) -> float:
    return (await predict_many([features]))[0]


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: (None if pd.isna(v) else float(v)) for k, v in zip(FEATS, row)}
            for row in df[FEATS].itertuples(index=False, name=None)]


# ---------------------------------------------------------------- rutas
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health", tags=["sistema"], summary="Estado del servicio")
def health():
    if STATE["latest"] is None:     # si falló al arrancar, reintenta aquí
        refresh_latest()
    if STATE["sales"] is None:
        refresh_sales()
    df, sales = STATE["latest"], STATE["sales"]
    return {"status": "ok" if df is not None else "sin_features",
            "features_rows": 0 if df is None else len(df),
            "sales_rows": 0 if sales is None else len(sales),
            "sources": STATE["source"],
            "endpoint": settings.serving_endpoint,
            "databricks_configured": db_config() is not None}


def _latest() -> pd.DataFrame:
    if STATE["latest"] is None and not refresh_latest():
        raise HTTPException(503, "La API no cargó las features (gold_latest). Revisa /health.")
    return STATE["latest"]


def _sales() -> pd.DataFrame:
    if STATE["sales"] is None and not refresh_sales():
        raise HTTPException(503, "La API no cargó las ventas históricas (gold_sales_daily). "
                                 "Corre el notebook 06 en Databricks y reinicia la API.")
    return STATE["sales"]


@app.get("/stores", tags=["catálogo"], response_model=list[str], summary="Lista de tiendas")
def stores():
    return sorted(_latest().index.get_level_values("store_id").unique().tolist())


@app.get("/items", tags=["catálogo"], response_model=list[str], summary="Productos de una tienda")
def items(store_id: str = Query(..., pattern=STORE_RE),
          q: str = Query("", max_length=30, description="Filtro por texto, ej. FOODS_3"),
          limit: int = Query(50, ge=1, le=1000)):
    df = _latest()
    if store_id not in df.index.get_level_values("store_id"):
        raise HTTPException(404, f"Tienda {store_id} no encontrada")
    ids = df.loc[store_id].index
    ids = ids[ids.str.contains(q.upper(), regex=False)] if q else ids
    return ids[:limit].tolist()


@app.post("/forecast", tags=["pronóstico"], response_model=RespuestaCobertura, summary="Pronóstico y cobertura de un producto")
async def forecast(req: SolicitudCobertura):
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

    return RespuestaCobertura(
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


@app.get("/sales", tags=["exploración"], response_model=RespuestaVentas, summary="Ventas históricas con filtros")
def sales(state_id: Literal["CA", "TX", "WI"] | None = Query(None, description="Filtrar por estado"),
          store_id: str | None = Query(None, pattern=STORE_RE, description="Filtrar por tienda"),
          cat_id: str | None = Query(None, pattern=CAT_RE, description="Filtrar por categoría"),
          date_from: date | None = Query(None, description="Desde (AAAA-MM-DD)"),
          date_to: date | None = Query(None, description="Hasta (AAAA-MM-DD)"),
          metric: Literal["units", "revenue"] = Query("units", description="Unidades o venta en $"),
          grain: Literal["day", "week", "month"] = Query("week", description="Agrupación temporal"),
          group_by: Literal["total", "state", "store", "category"] = Query("category",
                                                                           description="Una serie por…")):
    df = _sales()
    lo, hi = df["date"].min().date(), df["date"].max().date()
    date_from, date_to = date_from or lo, date_to or hi
    if date_from > date_to:
        raise HTTPException(422, "La fecha inicial es posterior a la final.")
    m = (df["date"] >= pd.Timestamp(date_from)) & (df["date"] <= pd.Timestamp(date_to))
    if state_id:
        m &= df["state_id"] == state_id
    if store_id:
        m &= df["store_id"] == store_id
    if cat_id:
        m &= df["cat_id"] == cat_id
    d = df[m]
    if d.empty:
        raise HTTPException(404, "No hay ventas con esos filtros (los datos van de "
                                 f"{lo} a {hi}).")

    col = {"state": "state_id", "store": "store_id", "category": "cat_id"}.get(group_by)
    freq = {"day": "D", "week": "W-MON", "month": "MS"}[grain]
    keys = [pd.Grouper(key="date", freq=freq, label="left", closed="left")] + ([col] if col else [])
    g = d.groupby(keys)[metric].sum().reset_index()
    g["series"] = g[col] if col else "Total"
    n_days = d["date"].nunique()
    return RespuestaVentas(
        metric=metric, grain=grain, group_by=group_by, date_from=date_from, date_to=date_to,
        total=round(float(d[metric].sum()), 2),
        avg_per_day=round(float(d[metric].sum()) / max(n_days, 1), 2),
        points=[PuntoVentas(period=max(p.date(), date_from), series=str(s), value=round(float(v), 2))
                for p, s, v in zip(g["date"], g["series"], g[metric])],
    )


def _group(df: pd.DataFrame, col: str) -> list[PronosticoGrupo]:
    g = df.groupby(col)[["last_28d_units", "forecast_28d_units"]].sum().reset_index()
    return [PronosticoGrupo(name=getattr(r, col), last_28d_units=round(r.last_28d_units, 1),
                          forecast_28d_units=round(r.forecast_28d_units, 1),
                          change_pct=round((r.forecast_28d_units / r.last_28d_units - 1) * 100, 1)
                          if r.last_28d_units > 0 else None)
            for r in g.itertuples(index=False)]


@app.get("/store-forecast", tags=["pronóstico"], response_model=RespuestaPronosticoTienda, summary="Pronóstico de todos los productos de una tienda")
async def store_forecast(store_id: str = Query(..., pattern=STORE_RE, examples=["CA_3"]),
                         refresh: bool = Query(False, description="Ignorar caché y volver a consultar el modelo")):
    """Pronóstico de los próximos 28 días para **todos** los productos de una tienda.
    Llama al endpoint por lotes de 1,000 filas y guarda el resultado 6 horas para no gastar cuota."""
    df = _latest()
    if store_id not in df.index.get_level_values("store_id"):
        raise HTTPException(404, f"Tienda {store_id} no encontrada")

    hit = CACHE.get(store_id)
    if hit and not refresh and time.time() - hit[0] < CACHE_TTL:
        return hit[1].model_copy(update={"cached": True})

    s = df.loc[store_id].reset_index()
    preds: list[float] = []
    for i in range(0, len(s), BATCH):
        preds += await predict_many(_records(s.iloc[i:i + BATCH]))

    s["forecast_28d_units"] = preds
    s["last_28d_units"] = (s["m28"].fillna(0) * HORIZON)
    s["change_units"] = s["forecast_28d_units"] - s["last_28d_units"]
    tot_last, tot_fc = float(s["last_28d_units"].sum()), float(s["forecast_28d_units"].sum())

    items_out = [PronosticoProducto(
        item_id=r.item_id, category=r.cat_id, dept=r.dept_id,
        last_28d_units=round(r.last_28d_units, 1), forecast_28d_units=round(r.forecast_28d_units, 1),
        change_units=round(r.change_units, 1),
        change_pct=round((r.forecast_28d_units / r.last_28d_units - 1) * 100, 1) if r.last_28d_units > 0 else None,
        price=None if pd.isna(r.price) else float(r.price)) for r in s.itertuples(index=False)]

    resp = RespuestaPronosticoTienda(
        store_id=store_id, as_of=s["origin"].iloc[0], n_items=len(s),
        last_28d_units=round(tot_last, 1), forecast_28d_units=round(tot_fc, 1),
        change_pct=round((tot_fc / tot_last - 1) * 100, 1) if tot_last > 0 else None,
        by_category=_group(s, "cat_id"), by_dept=_group(s, "dept_id"),
        items=items_out, cached=False, model_endpoint=settings.serving_endpoint)
    CACHE[store_id] = (time.time(), resp)
    return resp
