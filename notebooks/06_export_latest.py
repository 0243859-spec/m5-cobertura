# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Exportar `gold_latest` para la API
# MAGIC La API necesita las features "al día de hoy" de cada producto-tienda para armar la petición al endpoint.
# MAGIC Este notebook las publica como CSV en el Volume; la API lo descarga al arrancar con la **Files API** de Databricks (mismo token del endpoint).
# MAGIC Así la API no depende de un SQL Warehouse encendido (ahorra cuota y evita *cold starts*).

# COMMAND ----------
import os
C = "workspace.m5"
OUT_DIR = "/Volumes/workspace/m5/raw/exports"
os.makedirs(OUT_DIR, exist_ok=True)

pdf = spark.table(f"{C}.gold_latest").toPandas()
pdf.to_csv(f"{OUT_DIR}/gold_latest.csv", index=False)
print(f"{len(pdf):,} filas → {OUT_DIR}/gold_latest.csv")
