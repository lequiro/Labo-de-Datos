import numpy as np
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
#PAISES

PAISES = paises.copy()
PAISES = PAISES[['nombre',' iso3']]
PAISES = PAISES.rename(columns ={' iso3': 'iso3'})
PAISES['nombre'] = PAISES['nombre'].str.upper().replace({'Á': 'A','É': 'E','Í': 'I','Ó': 'O','Ú': 'U', 'Ô': 'O', 'Å':'A', '-': ' '}, regex = True)


#SECCIONES
lista_secciones_limpio = lista_secciones.copy()
lista_secciones_limpio = lista_secciones[['sede_id','sede_desc_castellano','tipo_seccion']]
lista_secciones_limpio = lista_secciones_limpio.rename(columns = {'sede_desc_castellano':'descripcion'})

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
flujos_monetarios = pd.DataFrame()
for (index,i) in enumerate(cols_flujos):
    x= flujos_nuevo[['iso3',i]].copy()
    x.loc[:, 'Fecha'] = cols_flujos_fechas.iloc[index][0]
    x = x.rename(columns={i : 'monto'}) 
    flujos_monetarios = pd.concat([flujos_monetarios, x])

#%%
###################################  REDES SOCIALES   ############################
lista_sedes_datos_extrac = lista_sedes_datos.copy()


############### Posicion 1 del vector en el que se separa redes sociales ###########
lista_sedes_datos_1 = lista_sedes_datos_extrac[['sede_id','redes_sociales']]
lista_sedes_datos_1.loc[:,'redes_sociales'] = lista_sedes_datos_1['redes_sociales'].str.split(' // ').str[0]
lista_sedes_datos_1 = lista_sedes_datos_1[~(lista_sedes_datos_1['redes_sociales'].isnull())].reset_index(drop= True)
lista_sedes_datos_1['tipo_red'] = (np.zeros(len(lista_sedes_datos_1))).astype('object')

#facebook
lista_indices_1 = np.where(lista_sedes_datos_1['redes_sociales'].str.contains('facebook', case=False))[0]
lista_sedes_datos_1.loc[lista_indices_1,'tipo_red'] = ['facebook']*len(lista_indices_1)
#twitter
lista_indices_2 = np.where(lista_sedes_datos_1['redes_sociales'].str.contains('twitter|@', case=False))[0]
lista_sedes_datos_1.loc[lista_indices_2,'tipo_red'] = ['twitter']*len(lista_indices_2)
#instagram
lista_indices_3 = np.where(lista_sedes_datos_1['redes_sociales'].str.contains('instagram|cscrs2018|embajadaargentinaenjapon', case=False))[0]
lista_sedes_datos_1.loc[lista_indices_3,'tipo_red'] = ['instagram']*len(lista_indices_3)

#asigno manualmente
lista_sedes_datos_1.loc[113,'tipo_red'] = 'linkedin'



############### Posicion 2 del vector en el que se separa redes sociales ###########
lista_sedes_datos_2 = lista_sedes_datos_extrac[['sede_id','redes_sociales']]
lista_sedes_datos_2.loc[:,'redes_sociales'] = lista_sedes_datos_2['redes_sociales'].str.split(' // ').str[1]
lista_sedes_datos_2 = lista_sedes_datos_2[~(lista_sedes_datos_2['redes_sociales'].isnull())].reset_index(drop= True)
lista_sedes_datos_2 = lista_sedes_datos_2[~(lista_sedes_datos_2['redes_sociales'] == ' ')].reset_index(drop= True)
lista_sedes_datos_2['tipo_red'] = (np.zeros(len(lista_sedes_datos_2))).astype('object')

#facebook
lista_indices_1 = np.where(lista_sedes_datos_2['redes_sociales'].str.contains('facebook', case=False))[0]
lista_sedes_datos_2.loc[lista_indices_1,'tipo_red'] = ['facebook']*len(lista_indices_1)
#twitter
lista_indices_2 = np.where(lista_sedes_datos_2['redes_sociales'].str.contains('twitter|@', case=False))[0]
lista_sedes_datos_2.loc[lista_indices_2,'tipo_red'] = ['twitter']*len(lista_indices_2)
#instagram
lista_indices_3 = np.where(lista_sedes_datos_2['redes_sociales'].str.contains('instagram| argenmozambique | consuladoargentinomia | arg_trinidad_tobago ', case=False))[0]
lista_sedes_datos_2.loc[lista_indices_3,'tipo_red'] = ['instagram']*len(lista_indices_3)
#youtube
lista_indices_4 = np.where(lista_sedes_datos_2['redes_sociales'].str.contains('youtube', case=False))[0]
lista_sedes_datos_2.loc[lista_indices_4,'tipo_red'] = ['instagram']*len(lista_indices_4)
#asigno a mano
lista_sedes_datos_2.loc[34,'tipo_red'] = 'linkedin'



################# Posicion 3 del vector en el que se separa redes sociales ############
lista_sedes_datos_3 = lista_sedes_datos_extrac[['sede_id','redes_sociales']]
lista_sedes_datos_3.loc[:,'redes_sociales']= lista_sedes_datos_3['redes_sociales'].str.split(' // ').str[2]
lista_sedes_datos_3 = lista_sedes_datos_3[~(lista_sedes_datos_3['redes_sociales'].isnull())].reset_index(drop= True)
lista_sedes_datos_3 = lista_sedes_datos_3[~(lista_sedes_datos_3['redes_sociales'] == ' ')].reset_index(drop= True)
lista_sedes_datos_3['tipo_red'] = (np.zeros(len(lista_sedes_datos_3))).astype('object')

#facebook
lista_indices_1 = np.where(lista_sedes_datos_3['redes_sociales'].str.contains('facebook', case=False))[0]
lista_sedes_datos_3.loc[lista_indices_1,'tipo_red'] = ['facebook']*len(lista_indices_1)
#twitter
lista_indices_2 = np.where(lista_sedes_datos_3['redes_sociales'].str.contains('twitter|@', case=False))[0]
lista_sedes_datos_3.loc[lista_indices_2,'tipo_red'] = ['twitter']*len(lista_indices_2)
#instagram
lista_indices_3 = np.where(lista_sedes_datos_3['redes_sociales'].str.contains('instagram| argentinaencolombia | argentinaenjamaica | Embajada  Argentina  en  Honduras ', case=False))[0]
lista_sedes_datos_3.loc[lista_indices_3,'tipo_red'] = ['instagram']*len(lista_indices_3)
#youtube
lista_indices_4 = np.where(lista_sedes_datos_3['redes_sociales'].str.contains('youtube', case=False))[0]
lista_sedes_datos_3.loc[lista_indices_4,'tipo_red'] = ['youtube']*len(lista_indices_4)


################# Posicion 4 del vector en el que se separa redes sociales ##############
lista_sedes_datos_4 = lista_sedes_datos_extrac[['sede_id','redes_sociales']]
lista_sedes_datos_4.loc[:,'redes_sociales'] = lista_sedes_datos_4['redes_sociales'].str.split(' // ').str[3]
lista_sedes_datos_4 = lista_sedes_datos_4[~(lista_sedes_datos_4['redes_sociales'].isnull())].reset_index(drop= True)
lista_sedes_datos_4 = lista_sedes_datos_4[~(lista_sedes_datos_4['redes_sociales'] == ' ')].reset_index(drop= True)
lista_sedes_datos_4['tipo_red'] = (np.zeros(len(lista_sedes_datos_4))).astype('object')

#instagram
lista_indices_3 = np.where(lista_sedes_datos_4['redes_sociales'].str.contains('instagram| Consulado  Argentino  en  Barcelona ', case=False))[0]
lista_sedes_datos_4.loc[lista_indices_3,'tipo_red'] = ['instagram']*len(lista_indices_3)
#youtube
lista_indices_4 = np.where(lista_sedes_datos_4['redes_sociales'].str.contains('youtube', case=False))[0]
lista_sedes_datos_4.loc[lista_indices_4,'tipo_red'] = ['youtube']*len(lista_indices_4)
#asigno a mano
lista_sedes_datos_4.loc[4,'tipo_red'] = 'flickr'



################################## concatenacion total ##################################
redes_sociales = concatenated_df = pd.concat([lista_sedes_datos_1, lista_sedes_datos_2, lista_sedes_datos_3, lista_sedes_datos_4], ignore_index=True)
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