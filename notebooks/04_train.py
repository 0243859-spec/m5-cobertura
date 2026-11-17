# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Entrenamiento — pronóstico directo de 28 días (LightGBM + MLflow)
# MAGIC - **Tarea:** regresión. Objetivo = unidades vendidas en los **próximos 28 días** por producto-tienda.
# MAGIC - **Métrica principal: WAPE** = Σ|real − pronóstico| / Σ real. El 59.8% de los días tienen venta 0, así que MAPE divide entre cero y no es usable; WAPE pondera por volumen, que es lo que importa para inventario.
# MAGIC - **Líneas base** (el modelo debe ganarles a ambas):
# MAGIC   1. *Naive reciente*: promedio de los últimos 28 días × 28 (la regla que usaría un planeador).
# MAGIC   2. *Año anterior*: lo vendido en el mismo periodo del año pasado.
# MAGIC - **Objetivo Tweedie:** para conteos con muchos ceros (demanda intermitente).
# MAGIC - **Validación temporal sin fuga:** `train` → `valid` (para early stopping, elegir `tweedie_variance_power` y el factor de sesgo) → `test` (último origen, solo para reportar). Ningún objetivo de entrenamiento se traslapa con el periodo evaluado.

# COMMAND ----------
# MAGIC %pip install lightgbm==4.5.0
# MAGIC %restart_python

# COMMAND ----------
import mlflow, lightgbm as lgb, numpy as np, pandas as pd
from datetime import timedelta
from pyspark.sql import functions as F
from mlflow.models import infer_signature

C = "workspace.m5"
MODEL_NAME = f"{C}.demand_model"
H = 28
mlflow.set_registry_uri("databricks-uc")
USER = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{USER}/m5_demand_forecast")

FEATS = ["item_id_idx","dept_id_idx","cat_id_idx","store_id_idx","state_id_idx",
         "m7","m14","m28","m56","m112","std28","zf28","zf112","days_since_sale","ly28",
         "price","price_norm","price_chg28","snap_days","event_days","month"]
CATS = ["dept_id_idx","cat_id_idx","store_id_idx","state_id_idx"]

g = spark.table(f"{C}.gold_train_direct")
test_origin  = g.agg(F.max("origin")).first()[0]          # último origen con 28 días completos
valid_origin = test_origin - timedelta(days=H)
print("valid:", valid_origin, "| test:", test_origin)

def load(cond):
    pdf = g.where(cond).select("id","origin", *FEATS, "target").toPandas()
    pdf[FEATS] = pdf[FEATS].astype("float32")
    return pdf

# train: orígenes cuyo horizonte termina antes del origen de validación
train = load(F.col("origin") <= F.lit(valid_origin - timedelta(days=H)))
valid = load(F.col("origin") == F.lit(valid_origin))
# refit: todo lo que termina antes del origen de prueba
refit = pd.concat([train, load((F.col("origin") > F.lit(valid_origin - timedelta(days=H)))
                               & (F.col("origin") <= F.lit(test_origin - timedelta(days=H))))], ignore_index=True)
test  = load(F.col("origin") == F.lit(test_origin))
print(f"train: {len(train):,}  valid: {len(valid):,}  refit: {len(refit):,}  test: {len(test):,}")

# COMMAND ----------
def wape(y, p): return float(np.abs(y - p).sum() / y.sum())

base_params = dict(objective="tweedie", learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_estimators=3000, verbose=-1)

# 1) Búsqueda de tweedie_variance_power con early stopping sobre validación
results = {}
for power in [1.1, 1.3, 1.5]:
    with mlflow.start_run(run_name=f"search_tweedie_{power}"):
        m = lgb.LGBMRegressor(**base_params, tweedie_variance_power=power)
        m.fit(train[FEATS], train["target"], categorical_feature=CATS,
              eval_set=[(valid[FEATS], valid["target"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = np.clip(m.predict(valid[FEATS]), 0, None)
        results[power] = dict(wape=wape(valid["target"], pv), iters=m.best_iteration_,
                              bias_factor=float(valid["target"].sum() / pv.sum()))
        mlflow.log_params({"tweedie_variance_power": power, "best_iteration": m.best_iteration_})
        mlflow.log_metric("valid_wape_28d", results[power]["wape"])
        print(f"power={power}: valid WAPE={results[power]['wape']:.4f}  árboles={m.best_iteration_}")

best_power = min(results, key=lambda k: results[k]["wape"])
best = results[best_power]
print("Mejor tweedie_variance_power:", best_power)

# COMMAND ----------
# 2) Modelo final: mejor hiperparámetro, reentrenado con todos los orígenes disponibles antes del test
params = {**base_params, "tweedie_variance_power": best_power, "n_estimators": int(best["iters"] * 1.1)}

with mlflow.start_run(run_name="lgbm_direct_28d") as run:
    model = lgb.LGBMRegressor(**params)
    model.fit(refit[FEATS], refit["target"], categorical_feature=CATS)
    pred = np.clip(model.predict(test[FEATS]), 0, None)

    y = test["target"].values
    naive = (test["m28"] * H).fillna(0).values
    ly = test["ly28"].fillna(test["m28"] * H).fillna(0).values

    parts = test["id"].str.split("_")
    ev = pd.DataFrame({"y": y, "p": pred, "n": naive,
                       "grp": parts.str[0] + "_" + parts.str[3] + "_" + parts.str[4]})
    sc = ev.groupby("grp")[["y","p","n"]].sum()

    metrics = {
        "wape_28d": wape(y, pred),
        "wape_28d_naive": wape(y, naive),
        "wape_28d_last_year": wape(y, ly),
        "mejora_vs_naive": 1 - wape(y, pred) / wape(y, naive),
        "wape_store_cat": wape(sc.y, sc.p),
        "wape_store_cat_naive": wape(sc.y, sc.n),
        "rmse_28d": float(np.sqrt(np.mean((y - pred) ** 2))),
        "bias": float(pred.sum() / y.sum() - 1),
        "bias_factor_valid": best["bias_factor"],
    }
    mlflow.log_params({**params, "n_features": len(FEATS), "horizon_days": H, "origin_step_days": 3})
    mlflow.log_metrics(metrics)

    sig = infer_signature(test[FEATS].head(100), pred[:100])
    mlflow.lightgbm.log_model(model.booster_, "model", signature=sig,
                              input_example=test[FEATS].head(3),
                              registered_model_name=MODEL_NAME)

for k, v in metrics.items(): print(f"{k:24s} {v:.4f}")

# COMMAND ----------
imp = pd.DataFrame({"feature": FEATS, "gain": model.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
display(imp)
