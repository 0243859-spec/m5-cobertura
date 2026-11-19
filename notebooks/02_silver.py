# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — limpieza, tipos y formato largo
# MAGIC | Problema | Decisión | Justificación |
# MAGIC |---|---|---|
# MAGIC | Ventas en formato ancho (1,941 columnas) | `unpivot` a 1 fila por producto-tienda-día | Formato necesario para modelar series de tiempo |
# MAGIC | Volumen (~59M filas) | Solo desde 2015-01-01 (~15M filas) | Cuota de Free Edition; ~16 meses bastan para estacionalidad anual |
# MAGIC | Tipos string de Bronze | Cast a date/int/double | Bronze guarda todo como texto |
# MAGIC | Ventas en 0 antes de que el producto tenga precio | Se eliminan | No es demanda cero: el producto aún no se vendía (cero estructural) |
# MAGIC | Nulos en eventos | `"none"` | Ausencia de evento es información, no dato faltante |
# MAGIC | Duplicados | `dropDuplicates` por llave | Garantiza 1 fila por producto-tienda-día |
# MAGIC | Outliers de venta | Se **marcan** (`is_outlier`, > p99.9 por producto); no se borran ni se modifican | Son picos reales (eventos, promociones) y forman parte de la demanda a pronosticar. Son solo 0.08 %; el modelo usa promedios de 7–112 días y objetivo Tweedie, que amortiguan picos aislados |

# COMMAND ----------
from pyspark.sql import functions as F, Window as W
C = "workspace.m5"
START = "2015-01-01"

cal = (spark.table(f"{C}.bronze_calendar")
       .select(F.to_date("date").alias("date"), F.col("wm_yr_wk").cast("int"), "weekday",
               F.col("wday").cast("int"), F.col("month").cast("int"), F.col("year").cast("int"), "d",
               F.coalesce("event_name_1", F.lit("none")).alias("event_name"),
               F.coalesce("event_type_1", F.lit("none")).alias("event_type"),
               *[F.col(f"snap_{s}").cast("int").alias(f"snap_{s}") for s in ["CA","TX","WI"]])
       .dropDuplicates(["date"]))
cal.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{C}.silver_calendar")

prices = (spark.table(f"{C}.bronze_sell_prices")
          .select("store_id","item_id", F.col("wm_yr_wk").cast("int"), F.col("sell_price").cast("double"))
          .where(F.col("sell_price") > 0)
          .dropDuplicates(["store_id","item_id","wm_yr_wk"]))
prices.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{C}.silver_sell_prices")

# COMMAND ----------
wide = spark.table(f"{C}.bronze_sales").drop("_ingested_at","_source_file")
# calendar.csv llega hasta d_1969 (incluye 28 días futuros sin ventas); solo usamos días que existen en sales
sales_days = set(c for c in wide.columns if c.startswith("d_"))
keep_days = [r.d for r in cal.where(F.col("date") >= START).orderBy("date").select("d").collect() if r.d in sales_days]
print(f"Días a usar: {len(keep_days)} ({keep_days[0]} → {keep_days[-1]})")
ids = ["id","item_id","dept_id","cat_id","store_id","state_id"]

long = (wide.select(*ids, *keep_days)
        .unpivot(ids, keep_days, "d", "sales")
        .withColumn("sales", F.col("sales").cast("int"))
        .join(cal.select("d","date","wm_yr_wk"), "d")
        # inner join con precios = elimina ceros estructurales (producto sin precio esa semana)
        .join(prices, ["store_id","item_id","wm_yr_wk"], "inner")
        .where(F.col("sales").isNotNull() & (F.col("sales") >= 0))
        .dropDuplicates(["id","date"]))

p999 = long.groupBy("item_id").agg(F.percentile_approx("sales", 0.999).alias("p999"))
long = (long.join(p999, "item_id")
        .withColumn("is_outlier", (F.col("sales") > F.col("p999")) & (F.col("p999") > 0))
        .drop("p999","d"))

(long.write.mode("overwrite").option("overwriteSchema", True)
     .partitionBy("state_id").saveAsTable(f"{C}.silver_sales"))

s = spark.table(f"{C}.silver_sales")
display(s.agg(F.count("*").alias("filas"), F.min("date"), F.max("date"),
              F.mean(F.col("sales").eqNullSafe(0).cast("int")).alias("pct_ceros"),
              F.sum(F.col("is_outlier").cast("int")).alias("outliers")))
