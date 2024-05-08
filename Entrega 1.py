import pandas as pd
from inline_sql import sql, sql_val
import os
path = r'C:\Users\Luis Quispe\Desktop\Labo_Datos\´TP\datasets'
os.chdir(path)
#%%
entradas = os.listdir(path)

flujos = pd.read_csv(entradas[0])
lista_secciones= pd.read_csv(entradas[1])
lista_sedes_datos = pd.read_csv(entradas[2], on_bad_lines= 'skip')
lista_sedes = pd.read_csv(entradas[3])
paises = pd.read_csv(entradas[4])

#%%
flujos_nuevo = flujos.copy()
flujos_nuevo = flujos.T.reset_index()
flujos_nuevo.columns = flujos_nuevo.iloc[0] #te pone el nombre de la columnas
flujos_nuevo = flujos_nuevo.drop(0) #elimina la primera fila, que son las columnas
flujos_nuevo = flujos_nuevo.rename(columns = {'indice_tiempo' : 'paises'})
flujos_nuevo = flujos_nuevo.fillna(0)
#%%
#limpieza de datasets

paises_limpio = paises.copy()
paises_limpio = paises_limpio[['nombre',' iso2']]

lista_secciones_limpio = lista_secciones.copy()
lista_secciones_limpio = lista_secciones[['sede_id','sede_desc_castellano','tipo_seccion']]


lista_sedes_datos_limpio = lista_sedes_datos.copy()
lista_sedes_datos_limpio['redes_sociales'] = lista_sedes_datos_limpio['redes_sociales'].str.split(' // ')
#%%
#lista sedes
#Voy a eliminar los campos de mi tabla donde el estado sea "Inactivo"
lista_sedes_limpio = lista_sedes.copy()

for x in lista_sedes_limpio.index: 
  if lista_sedes_limpio.loc[x, "estado"] == 'Inactivo':
    lista_sedes_limpio.drop(x, inplace = True)


#Como se que son todas sedes activas, puedo eliminar la columna "estado"
#Al mismo tiempor quiero eliminar la columna de "sede_tipo" ya que no es relevante para mi objetivo
lista_sedes_limpio = lista_sedes_limpio[['sede_id','pais_iso_3','sede_tipo']]


#%%
