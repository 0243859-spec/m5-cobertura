# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — ingesta cruda
# MAGIC Lee los 3 CSV del Volume tal cual vienen de Kaggle y los guarda como Delta.
# MAGIC **Regla de Bronze:** no se limpia nada; solo se agregan columnas de auditoría.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG, SCHEMA = "workspace", "m5"
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

def ingest(file, table):
    df = (spark.read.option("header", True).option("inferSchema", False)  # todo como string en bronze
          .csv(f"{RAW}/{file}")
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.lit(file)))
    df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_{table}")
    print(f"bronze_{table}: {df.count():,} filas, {len(df.columns)} columnas")

ingest("calendar.csv", "calendar")
ingest("sell_prices.csv", "sell_prices")
ingest("sales_train_evaluation.csv", "sales")   # formato ancho: 1 columna por día (d_1..d_1941)
