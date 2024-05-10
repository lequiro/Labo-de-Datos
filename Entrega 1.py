import numpy as np
import pandas as pd
from inline_sql import sql, sql_val
import os
path = r'C:\Users\Luis Quispe\Desktop\Labo_Datos\´TP\datasets'
os.chdir(path)

#%%
def process_redes_sociales(lista,posicion, patrones, asignaciones):
    lista_sedes_datos_temp = lista[['sede_id','redes_sociales']]
    lista_sedes_datos_temp.loc[:,'redes_sociales'] = lista_sedes_datos_temp['redes_sociales'].str.split(' // ').str[posicion]
    lista_sedes_datos_temp = lista_sedes_datos_temp.dropna().reset_index(drop=True)
    lista_sedes_datos_temp = lista_sedes_datos_temp[~(lista_sedes_datos_temp['redes_sociales'] == ' ')].reset_index(drop=True)
    lista_sedes_datos_temp.loc[:,'tipo_red'] = np.zeros(len(lista_sedes_datos_temp), dtype='object')

    for i, patron in enumerate(patrones):
        lista_indices = np.where(lista_sedes_datos_temp['redes_sociales'].str.contains(patron, case=False))[0]
        lista_sedes_datos_temp.loc[lista_indices, 'tipo_red'] = [asignaciones[i]] * len(lista_indices)

    return lista_sedes_datos_temp


#%%
entradas = os.listdir(path)

flujos = pd.read_csv(entradas[0])
lista_secciones= pd.read_csv(entradas[1])
lista_sedes_datos = pd.read_csv(entradas[2], on_bad_lines= 'skip')
lista_sedes = pd.read_csv(entradas[3])
paises = pd.read_csv(entradas[4])



#%%
#PAISES

PAISES = paises.copy()
PAISES = PAISES[['nombre',' iso3']]
PAISES = PAISES.rename(columns ={' iso3': 'iso3'})
PAISES['nombre'] = PAISES['nombre'].str.upper().replace({'Á': 'A','É': 'E','Í': 'I','Ó': 'O','Ú': 'U', 'Ô': 'O', 'Å':'A', '-': ' '}, regex = True)

#SECCIONES
SECCIONES = lista_secciones.copy()
SECCIONES = lista_secciones[['sede_id','sede_desc_castellano','tipo_seccion']]
SECCIONES = SECCIONES.rename(columns = {'sede_desc_castellano':'descripcion'})

#%%
#FLUJOS MONETARIOS
flujos_nuevo = flujos.copy()
flujos_nuevo = flujos.T.reset_index()
flujos_nuevo.columns = flujos_nuevo.iloc[0] #te pone el nombre de la columnas
flujos_nuevo = flujos_nuevo.drop(0) #elimina la primera fila, que son las columnas
flujos_nuevo = flujos_nuevo.rename(columns = {'indice_tiempo' : 'nombre'})
flujos_nuevo = flujos_nuevo.fillna(0)
flujos_nuevo['nombre'] = flujos_nuevo['nombre'].str.upper().str.replace('_', ' ')

paises_reemplazar = {'VIET NAM': 'VIETNAM',
                    'BELARUS' :'BIELORRUSIA',
                    'BOTSWANA' : 'BOTSUANA',
                    'ESTADO DE PALESTINA' : 'PALESTINA',
                    'GUAYANA' : 'GUYANA',
                    'REPUBLICA DE MOLDOVA': 'MOLDAVIA',
                    'SURINAME': 'SURINAM',
                    'TRINIDAD TOBAGO': 'TRINIDAD Y TOBAGO',
                    'TFYR DE MACEDONIA': 'MACEDONIA',
                    'SERBIA MONTENEGRO': 'MONTENEGRO',
                    'SANTO TOME PRINCIPE': 'SANTO TOME Y PRINCIPE',
                    'REP DEM DEL CONGO': 'REPUBLICA DEMOCRATICA DEL CONGO',
                    'REP DE COREA' : 'COREA DEL SUR',
                    'POLINESIA FRANCES': 'POLINESIA FRANCESA',
                    'MEJICO': 'MEXICO',
                    'LAO': 'LAOS',
                    'KAZAKHSTAN': 'KAZAJISTAN',
                    'JORDAN': 'JORDANIA',
                    'KATAR': 'QATAR',
                    'FEDERACION RUSA': 'RUSIA',
                    'ESPANNA': 'ESPAÑA',
                    'ESTADOS UNIDOS': 'ESTADOS UNIDOS DE AMERICA',
                    'CZECHIA':'REPUBLICA CHECA',
                    'COTE DIVOIRE': 'COSTA DE MARFIL',
                    'CONGO': 'REPUBLICA DEL CONGO',
                    'CHINA PROVINCIA DE TAIWAN': 'TAIWAN',
                    'CHINA RAE DE HONG KONG': 'HONG KONG',
                    'CHINA RAE DE MACAO': 'MACAO',
                    'BRUNEI DARUSSALAM': 'BRUNEI',
                    'BOSNIA HERZEGOVINA': 'BOSNIA Y HERZEGOVINA'
                    }
flujos_nuevo = flujos_nuevo.replace(paises_reemplazar)
flujos_nuevo = flujos_nuevo.drop([50,114])
flujos_nuevo = pd.merge(flujos_nuevo, PAISES, how='inner')


####################### CREACION FLUJOS MONETARIOS ###########################
cols_flujos =flujos_nuevo.columns[39:44]
cols_flujos_fechas = cols_flujos.str.extract('(\d+)-')
FLUJOS_MONETARIOS = pd.DataFrame()
for (index,i) in enumerate(cols_flujos):
    x= flujos_nuevo[['iso3',i]].copy()
    x.loc[:, 'fecha'] = cols_flujos_fechas.iloc[index][0]
    x = x.rename(columns={i : 'monto'}) 
    FLUJOS_MONETARIOS = pd.concat([FLUJOS_MONETARIOS, x])

#%%
###################################  REDES SOCIALES   ############################
lista_sedes_datos_extrac = lista_sedes_datos.copy()


############### Posicion 1 del vector en el que se separa redes sociales ###########
patrones_1 = ['facebook', 'twitter|@', 'instagram|cscrs2018|embajadaargentinaenjapon']
asignaciones_1 = ['facebook', 'twitter', 'instagram']
lista_sedes_datos_1 = process_redes_sociales(lista_sedes_datos, 0, patrones_1, asignaciones_1)
lista_sedes_datos_1.loc[113,'tipo_red'] = 'linkedin'

############### Posicion 2 del vector en el que se separa redes sociales ###########
patrones_2 = ['facebook', 'twitter|@', 'instagram| argenmozambique | consuladoargentinomia | arg_trinidad_tobago ', 'youtube']
asignaciones_2 = ['facebook', 'twitter', 'instagram', 'youtube']
lista_sedes_datos_2 = process_redes_sociales(lista_sedes_datos, 1, patrones_2, asignaciones_2)
lista_sedes_datos_2.loc[34,'tipo_red'] = 'linkedin'


################# Posicion 3 del vector en el que se separa redes sociales ############

patrones_3 = ['facebook', 'twitter|@', 'instagram| argentinaencolombia | argentinaenjamaica | Embajada  Argentina  en  Honduras ', 'youtube']
asignaciones_3 = ['facebook', 'twitter', 'instagram', 'youtube']
lista_sedes_datos_3 = process_redes_sociales(lista_sedes_datos, 2, patrones_3, asignaciones_3)


################ Posicion 4 del vector en el que se separa redes sociales ##############
patrones_4 = ['instagram| Consulado  Argentino  en  Barcelona ', 'youtube']
asignaciones_4 = ['instagram', 'youtube']
lista_sedes_datos_4 = process_redes_sociales(lista_sedes_datos, 3, patrones_4, asignaciones_4)
lista_sedes_datos_4.loc[4,'tipo_red'] = 'flickr'

################################## concatenacion total ##################################
REDES_SOCIALES = pd.concat([lista_sedes_datos_1,
                            lista_sedes_datos_2,
                            lista_sedes_datos_3,
                            lista_sedes_datos_4],
                            ignore_index=True)
REDES_SOCIALES = REDES_SOCIALES.rename(columns = {'redes_sociales': 'contacto'})
#%%
################################## SEDES ##################################
#Voy a eliminar los campos de mi tabla donde el estado sea "Inactivo"
SEDES = lista_sedes.copy()

for x in SEDES.index: 
  if SEDES.loc[x, "estado"] == 'Inactivo':
    SEDES.drop(x, inplace = True)


#Como se que son todas sedes activas, puedo eliminar la columna "estado"
#Al mismo tiempo quiero eliminar la columna de "sede_tipo" ya que no es relevante para mi objetivo
SEDES = SEDES[['sede_id','pais_iso_3','sede_tipo']]
SEDES = SEDES.rename(columns = {'pais_iso_3' : 'iso3'})