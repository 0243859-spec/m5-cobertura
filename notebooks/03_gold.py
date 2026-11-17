# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — features para pronóstico directo de 28 días
# MAGIC **Enfoque directo:** para cada producto-tienda y una fecha de corte (*origen*), se calculan features con **todo lo conocido hasta el día anterior** y el objetivo es la **venta total de los 28 días siguientes**.
# MAGIC Es exactamente la pregunta que responde la app de cobertura: *¿cuánto voy a vender en los próximos 28 días?*
# MAGIC
# MAGIC | Tabla | Contenido |
# MAGIC |---|---|
# MAGIC | `gold_train_direct` | Un registro por producto-tienda-origen (orígenes cada 3 días) con features + objetivo. Para entrenar. |
# MAGIC | `gold_latest` | Features al día siguiente del último dato (el "hoy" de la app). Lo consulta la API. |
# MAGIC | `gold_sales_daily` | Ventas y venta en $ por día, tienda y categoría. Para el dashboard. |
# MAGIC
# MAGIC **Sin fuga de información:** todas las ventanas de features terminan el día **anterior** al origen (`rangeBetween(-k, -1)`); el calendario del horizonte (SNAP, eventos) sí se usa porque se conoce por adelantado.
# MAGIC **Codificación:** categóricas → índice entero (LightGBM las maneja nativo). **Normalización:** precio relativo a su histórico.

# COMMAND ----------
from pyspark.sql import functions as F, Window as W
from datetime import timedelta
C = "workspace.m5"
H = 28            # horizonte
STEP = 3          # un origen de entrenamiento cada 3 días
MIN_HIST = 112    # días de historia mínima antes del primer origen

s = spark.table(f"{C}.silver_sales")
cal = spark.table(f"{C}.silver_calendar")
max_date = s.agg(F.max("date")).first()[0]
latest_origin = max_date + timedelta(days=1)
D0 = s.agg(F.min("date")).first()[0]
print("Último dato:", max_date, "| origen 'hoy' para la app:", latest_origin)

# Fila vacía por serie en el origen "hoy", para calcular sus features con las mismas ventanas
ids = s.select("id","item_id","dept_id","cat_id","store_id","state_id").distinct()
latest_rows = ids.withColumn("date", F.lit(latest_origin)).withColumn("sales", F.lit(None).cast("int")) \
                 .withColumn("sell_price", F.lit(None).cast("double"))
daily = (s.select("id","item_id","dept_id","cat_id","store_id","state_id","date","sales","sell_price")
          .unionByName(latest_rows)
          .withColumn("dn", F.datediff("date", F.lit(D0))))

w = W.partitionBy("id").orderBy("dn")
def past(k):   return w.rangeBetween(-k, -1)
def fut():     return w.rangeBetween(0, H - 1)

f = (daily
     .withColumn("m7",   F.avg("sales").over(past(7)))
     .withColumn("m14",  F.avg("sales").over(past(14)))
     .withColumn("m28",  F.avg("sales").over(past(28)))
     .withColumn("m56",  F.avg("sales").over(past(56)))
     .withColumn("m112", F.avg("sales").over(past(112)))
     .withColumn("std28", F.stddev("sales").over(past(28)))
     .withColumn("zf28",  F.avg((F.col("sales") == 0).cast("double")).over(past(28)))
     .withColumn("zf112", F.avg((F.col("sales") == 0).cast("double")).over(past(112)))
     .withColumn("last_sale_dn", F.max(F.when(F.col("sales") > 0, F.col("dn"))).over(w.rangeBetween(W.unboundedPreceding, -1)))
     .withColumn("days_since_sale", F.coalesce(F.col("dn") - F.col("last_sale_dn"), F.lit(999)))
     .withColumn("ly28", F.when(F.count("sales").over(w.rangeBetween(-364, -337)) == 28,
                                F.sum("sales").over(w.rangeBetween(-364, -337))))
     .withColumn("price", F.last("sell_price", ignorenulls=True).over(w.rangeBetween(W.unboundedPreceding, 0)))
     .withColumn("price_norm", F.col("price") / F.avg("sell_price").over(w.rangeBetween(W.unboundedPreceding, -1)))
     .withColumn("price_28ago", F.last("sell_price", ignorenulls=True).over(w.rangeBetween(W.unboundedPreceding, -28)))
     .withColumn("price_chg28", F.col("price") / F.col("price_28ago") - 1)
     .withColumn("hist_days", F.count("sales").over(w.rangeBetween(W.unboundedPreceding, -1)))
     # objetivo: venta total de los próximos 28 días (solo si los 28 días existen)
     .withColumn("target", F.when(F.count("sales").over(fut()) == H, F.sum("sales").over(fut())))
     .drop("last_sale_dn","price_28ago"))

# Calendario del horizonte (se conoce por adelantado): días SNAP por estado y días con evento
wc = W.orderBy("date").rowsBetween(0, H - 1)
calh = (cal.withColumn("event_flag", (F.col("event_type") != "none").cast("int"))
           .withColumn("event_days", F.sum("event_flag").over(wc))
           .withColumn("snap_CA_days", F.sum("snap_CA").over(wc))
           .withColumn("snap_TX_days", F.sum("snap_TX").over(wc))
           .withColumn("snap_WI_days", F.sum("snap_WI").over(wc))
           .select("date","event_days","snap_CA_days","snap_TX_days","snap_WI_days", F.month("date").alias("month")))

f = (f.join(calh, "date", "left")
      .withColumn("snap_days", F.expr("CASE state_id WHEN 'CA' THEN snap_CA_days WHEN 'TX' THEN snap_TX_days ELSE snap_WI_days END"))
      .drop("snap_CA_days","snap_TX_days","snap_WI_days"))

for c in ["item_id","dept_id","cat_id","store_id","state_id"]:
    idx = ids.select(c).distinct().withColumn(f"{c}_idx", F.dense_rank().over(W.orderBy(c)) - 1)
    f = f.join(F.broadcast(idx), c)

FEATS = ["item_id_idx","dept_id_idx","cat_id_idx","store_id_idx","state_id_idx",
         "m7","m14","m28","m56","m112","std28","zf28","zf112","days_since_sale","ly28",
         "price","price_norm","price_chg28","snap_days","event_days","month"]
KEYS = ["id","item_id","dept_id","cat_id","store_id","state_id", F.col("date").alias("origin")]

# COMMAND ----------
# Orígenes: cada STEP días + los dos orígenes exactos de evaluación (prueba = últimos 28 días; validación = 28 antes)
test_origin = max_date - timedelta(days=H - 1)
valid_origin = test_origin - timedelta(days=H)
is_origin = (F.col("dn") % STEP == 0) | F.col("date").isin(test_origin, valid_origin)
train_direct = (f.where((F.col("dn") >= MIN_HIST) & is_origin & F.col("target").isNotNull()
                        & (F.col("hist_days") >= 28))
                 .select(*KEYS, *FEATS, "target"))
train_direct.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{C}.gold_train_direct")

latest = f.where(F.col("date") == F.lit(latest_origin)).select(*KEYS, *FEATS)
latest.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{C}.gold_latest")

# Tabla de negocio para el dashboard
(s.groupBy("date","state_id","store_id","cat_id")
  .agg(F.sum("sales").alias("units"), F.sum(F.col("sales") * F.col("sell_price")).alias("revenue"))
  .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{C}.gold_sales_daily"))

t = spark.table(f"{C}.gold_train_direct")
print(f"gold_train_direct: {t.count():,} filas, {t.select('origin').distinct().count()} orígenes")
print(f"gold_latest: {spark.table(f'{C}.gold_latest').count():,} productos-tienda")
display(spark.table(f"{C}.gold_latest").limit(5))
