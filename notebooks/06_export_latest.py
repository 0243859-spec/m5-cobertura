# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Exportar tablas Gold para la API
# MAGIC | Archivo | Para qué lo usa la API |
# MAGIC |---|---|
# MAGIC | `gold_latest.csv` | Features "al día de hoy" de cada producto-tienda → petición al endpoint (pestañas Cobertura y Pronóstico por tienda) |
# MAGIC | `gold_sales_daily.csv` | Ventas diarias por tienda y categoría → pestaña Ventas (gráficas con filtros) |
# MAGIC
# MAGIC La API los descarga al arrancar con la **Files API** de Databricks (mismo token del endpoint),
# MAGIC así no depende de un SQL Warehouse encendido (ahorra cuota y evita *cold starts*).

# COMMAND ----------
import os
C = "workspace.m5"
OUT_DIR = "/Volumes/workspace/m5/raw/exports"
os.makedirs(OUT_DIR, exist_ok=True)

for table in ["gold_latest", "gold_sales_daily"]:
    pdf = spark.table(f"{C}.{table}").toPandas()
    pdf.to_csv(f"{OUT_DIR}/{table}.csv", index=False)
    print(f"{table}: {len(pdf):,} filas → {OUT_DIR}/{table}.csv")
