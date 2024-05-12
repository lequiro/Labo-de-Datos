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

REGION = (lista_sedes_datos[['pais_iso_3', 'region_geografica' ]]).unique()
REGION = REGION.rename(columns = {'pais_iso_3' : 'iso_3'})

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
cols_flujos =flujos_nuevo.columns[1:]
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
SEDES = SEDES[['sede_id','pais_iso_3']]
SEDES = SEDES.rename(columns = {'pais_iso_3' : 'iso3'})

#%%% CONSULTAS SQL


"""
PUNTO I)
Para cada país informar cantidad de sedes, cantidad de secciones en promedio que poseen 
sus sedes y el flujo monetario neto de Inversión Extranjera Directa (IED) del país en 
el año 2022. El orden del reporte debe respetar la cantidad de sedes (de manera 
descendente). En caso de empate, ordenar alfabéticamente por nombre de país

"""
#Cuento cantidad de sedes por pais
consulta1 = """
               SELECT p.iso3, COUNT(s.sede_id) AS cantidad_sedes,
               FROM PAISES AS p
               LEFT OUTER JOIN SEDES AS s ON p.iso3 = s.iso3
               GROUP BY p.iso3;

              """
              
tablaContadorSedes = sql^ consulta1
#Agrego resultado anterior a la tabla
consulta2 = """
               SELECT DISTINCT p.nombre, t.cantidad_sedes, p.iso3
               FROM PAISES AS p
                JOIN tablaContadorSedes AS t
               ON p.iso3 = t.iso3
              """

tablaPaisesYSedes = sql^ consulta2

#Junto tabla secciones y tabla sedes

consulta3 = """
   SELECT DISTINCT sec.tipo_seccion, sed.iso3, sed.sede_id,
   FROM SEDES AS sed
   LEFT JOIN SECCIONES AS sec
   ON sed.sede_id = sec.sede_id
    
"""
tablaSeccionesYSedes = sql^ consulta3
#Cuento cantidad de secciones por sede_id (cuento las veces que se repiten sede_id, ya que si sede_id se repite 2 veces significa que tiene dos secciones)
#FALTA CALCULAR EL PROMEDIO
consulta5 = """
            SELECT DISTINCT COUNT(sede_id) AS cantidad_secciones, sede_id, ANY_VALUE(iso3) AS iso3
            FROM tablaSeccionesYSedes 
            GROUP BY sede_id

    """
tablaCantidadSeccionesPorSede = sql^ consulta5

consulta4 = """
    SELECT ROUND(AVG(t.cantidad_secciones),2) AS promedio_secciones, ANY_VALUE(t.sede_id) AS sede_id, t.iso3
    FROM tablaCantidadSeccionesPorSede as t
    GROUP BY iso3
"""
tablaPromedioSeccionesPorSede = sql^ consulta4

consulta6 = """
        SELECT tc.sede_id, tp.cantidad_sedes, tc.promedio_secciones, tp.nombre, tp.iso3
        FROM tablaPaisesYSedes AS tp
        LEFT JOIN  tablaPromedioSeccionesPorSede AS tc
        ON tc.iso3 = tp.iso3
"""
tablaCantidadSedesYSeccionesPorPais = sql^ consulta6
#calculo el flujo en 2022 por pais
#PODRIAMOS AGREGAR EN QUE SE ESTA MIDIENDO EL FLUJO (USD$??)
consulta7 = """
    SELECT monto AS IED_2022, iso3
    FROM FLUJOS_MONETARIOS
    WHERE fecha == '2022'
"""

tablaIED2022 = sql^consulta7

#Uno este resultado con los resultados anteriores de cant de sedes por pais y secciones por sede y ordeno 

consulta8 = """
    SELECT tc.cantidad_sedes AS sedes, tc.nombre AS pais, ti.IED_2022, tc.promedio_secciones AS seccion
    FROM tablaCantidadSedesYSeccionesPorPais as tc
    LEFT JOIN tablaIED2022 as ti
    ON tc.iso3 = ti.iso3
    ORDER BY tc.cantidad_sedes DESC, tc.nombre
"""
tablaResultado1 = sql^ consulta8
#FALTA REEMPLAZAR LOS NULLS POR 0 SI ES FLOAT O "-" SI SE ESPERABA UN STRING
        
"""
III) Para saber cuál es la vía de comunicación de las sedes en cada país, nos
hacemos la siguiente pregunta: ¿Cuán variado es, en cada el país, el tipo de 
redes sociales que utilizan las sedes? Se espera como respuesta que para cada país 
se informe la cantidad de tipos de redes distintas utilizadas. Por ejemplo, si en Chile u
tilizan 4 redes de facebook, 5 de instagram y 4 de twitter, el valor para Chile debería ser 3 
(facebook, instagram y twitter).
"""
#armo tabla de sedes con red social para lograr tener la foreign key iso3

consulta9 = """
    SELECT rs.contacto, rs.tipo_red, s.iso3
    FROM REDES_SOCIALES AS rs
    LEFT JOIN SEDES as s
    ON s.sede_id = rs.sede_id
"""

tablaSedeYRedSocial = sql^consulta9

#luego conecto el resultado anterior con la tabla paises
consulta10 = """
    SELECT *
    FROM PAISES as p
    LEFT OUTER JOIN tablaSedeYRedSocial as t
    ON p.iso3 = t.iso3
"""
tablaPaisesYRedes = sql ^ consulta10

#cuento la cantidad de tipo_red para cada pais, uso el distinct para que me cuente solo los que son distintos


consulta11 = """
        SELECT COUNT(DISTINCT tipo_red) AS cantidad_tipos, iso3, ANY_VALUE(nombre) AS pais
        FROM tablaPaisesYRedes
        GROUP BY iso3
"""

tablaCantidadRedPorPais = sql ^ consulta11
