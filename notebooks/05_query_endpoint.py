# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Prueba del endpoint `m5-demand`
# MAGIC Toma productos-tienda de `gold_latest` (features "al día de hoy"), los manda al endpoint publicado y muestra el pronóstico de 28 días.
# MAGIC También imprime el **JSON de ejemplo** para consultar el endpoint desde fuera de Databricks (curl / Postman / la API).

# COMMAND ----------
import json
from mlflow.deployments import get_deploy_client

C = "workspace.m5"
ENDPOINT = "m5-demand"
FEATS = ["item_id_idx","dept_id_idx","cat_id_idx","store_id_idx","state_id_idx",
         "m7","m14","m28","m56","m112","std28","zf28","zf112","days_since_sale","ly28",
         "price","price_norm","price_chg28","snap_days","event_days","month"]

rows = (spark.table(f"{C}.gold_latest")
        .where("id IN ('FOODS_3_090_CA_3_evaluation','HOBBIES_1_001_CA_1_evaluation','HOUSEHOLD_1_118_TX_2_evaluation')")
        .toPandas())
records = [{k: (None if v != v else float(v)) for k, v in r.items()} for r in rows[FEATS].to_dict("records")]

client = get_deploy_client("databricks")          # dentro de Databricks no necesita token
resp = client.predict(endpoint=ENDPOINT, inputs={"dataframe_records": records})

out = rows[["id","m28"]].assign(venta_ult_28d=lambda d: (d.m28 * 28).round(0),
                                pronostico_28d=[round(p, 1) for p in resp["predictions"]])
display(out)

# COMMAND ----------
# JSON de ejemplo para consultar desde fuera (README / entrega)
print(json.dumps({"dataframe_records": records[:1]}, indent=2))
