from meteostat import Point, Daily, Hourly
from datetime import datetime
import pandas as pd

# -----------------------
# PARÁMETROS DEL SCRIPT
# -----------------------

# Ubicación: Bilbao / Sondica
bilbao = Point(43.3, -2.9333)

# Rango de fechas
start = datetime(2000, 1, 1)
end = datetime(2024, 12, 31)

# ------------------------
# DATOS DIARIOS
# ------------------------

print("Descargando datos diarios...")
daily_data = Daily(bilbao, start, end)
daily_df = daily_data.fetch()

# Guardar CSV
daily_df.to_csv("bilbao_diario_2000_2024.csv")
print("✅ Datos diarios guardados como 'bilbao_diario_2000_2024.csv'")

# ------------------------
# DATOS HORARIOS
# ------------------------

print("Descargando datos horarios (esto puede tardar)...")
hourly_data = Hourly(bilbao, start, end)
hourly_df = hourly_data.fetch()

# Guardar CSV
hourly_df.to_csv("bilbao_horario_2000_2024.csv")
print("✅ Datos horarios guardados como 'bilbao_horario_2000_2024.csv'")
