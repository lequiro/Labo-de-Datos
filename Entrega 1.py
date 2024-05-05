import pandas as pd
from inline_sql import sql, sql_val


tabla_sedes = "/Users/juanaotermin/Documents/Facultad Exactas/Laboratorio de Datos/TP_01/lista-sedes.csv"

df = pd.read_csv(tabla_sedes, index_col= 'sede_id')

#Voy a eliminar los campos de mi tabla donde el estado sea "Inactivo"

for x in df.index: 
  if df.loc[x, "estado"] == 'Inactivo':
    df.drop(x, inplace = True)


print(df.info())

#Como se que son todas sedes activas, puedo eliminar la columna "estado"
#Al mismo tiempor quiero eliminar la columna de "sede_tipo" ya que no es relevante para mi objetivo

df_limpio = df.drop(["estado", "sede_tipo"], axis = 1)
#%%
