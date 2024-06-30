"""
Trabajo Práctico 01
Materia: Laboratorio de datos - FCEyN - UBA
Integrantes: Otermín Juana, Quispe Rojas Luis Enrique , Vilcovsky Maia
Este codigo contiene funciones, aplicaciones en Pandas, consultas en SQL y visualizacion de datos a partir de graficos.

Fecha  : 2024-05-12
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns  
from inline_sql import sql, sql_val
#%%
#Esto te abre los gráficos en una ventana aparte (CORRERLO ES OPCIONAL)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
#%%
#seteo el working directory
desktop_directory = os.path.join(os.path.expanduser('~'), "Desktop")
desktop_directory = os.path.join(desktop_directory,"TP01-MLJ", "TablasOriginales")
os.chdir(desktop_directory)
entradas = os.listdir(desktop_directory)
#%%
def process_redes_sociales(lista,posicion, patrones, asignaciones):
    """
     Procesa los datos de redes sociales de una lista de sedes, extrayendo información específica
     según un patrón dado y asignando un tipo de red social a cada entrada.
    
     Parámetros:
     - lista: DataFrame. Información sobre sedes y sus redes sociales.
     - posicion: int. Posición de la red social dentro de la lista separada por '//' que se debe extraer.
     - patrones: list of strings. Patrones para identificar tipos específicos de redes sociales.
     - asignaciones: list of strings. Nombres correspondientes a los tipos de redes sociales identificados
       por los patrones.
    
     Retorna:
     DataFrame. Datos procesados de las redes sociales de las sedes, incluyendo el tipo de red social asignado.
    
     Comportamiento:
     1. Selecciona 'sede_id' y 'redes_sociales' del DataFrame de entrada.
     2. Extrae la red social en la posición especificada.
     3. Elimina filas nulas y redes sociales en blanco.
     4. Crea 'tipo_red' con valores predeterminados.
     5. Asigna tipos de red social según patrones.
     6. Retorna el DataFrame con datos procesados.
     """
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
flujos = pd.read_csv(entradas[0])
lista_secciones= pd.read_csv(entradas[1])
lista_sedes_datos = pd.read_csv(entradas[2], on_bad_lines= 'skip')
lista_sedes = pd.read_csv(entradas[3])
paises = pd.read_csv(entradas[4])
#%%
'''
SECCION PAISES/SECCIONES/REGIONES
En esta sección creamos las tablas PAISES, SECCIONES Y REGION modificando los valores de columnas
y filas según el objetivo del trabajo
'''
#PAISES
PAISES = paises.copy()
PAISES = PAISES[['nombre',' iso3']]
PAISES = PAISES.rename(columns ={' iso3': 'iso3'})
PAISES['nombre'] = PAISES['nombre'].str.upper().replace({'Á': 'A','É': 'E','Í': 'I','Ó': 'O','Ú': 'U', 'Ô': 'O', 'Å':'A', '-': ' '}, regex = True)
#SECCIONES
SECCIONES = lista_secciones.copy()
SECCIONES = lista_secciones[['sede_id','sede_desc_castellano','tipo_seccion']]
SECCIONES = SECCIONES.rename(columns = {'sede_desc_castellano':'descripcion'})
#REGIONES
REGION = lista_sedes_datos[['pais_iso_3', 'region_geografica' ]]
REGION = REGION.rename(columns = {'pais_iso_3' : 'iso3'})
REGION = REGION.drop_duplicates()



#####################################PUNTO F(GQM)#########################################
proporcion_not_cellphone = lista_secciones['telefono_principal'].isna().sum() / len(lista_secciones['telefono_principal']) * 100
##########################################################################################

#%%
'''
SECCION FLUJOS MONETARIOS
Se realiza una transposición de flujos, se renombran las columnas, se elimina la primera fila,
se reemplazan valores nulos por 0, se convierten los nombres a mayúsculas y se reemplazan ciertos
nombres de países por sus equivalentes en la tabla paises.
'''
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
'''
SECCION REDES SOCIALES
Dado la funcion process_redes_sociales  genera un dataframe a partir de lista_sedes_datos que 
tiene por columnas ['sede_id', 'contacto', 'tipo red']

'''
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

#####################################PUNTO F(GQM)#########################################
proporcion_not_URL = ((~REDES_SOCIALES['contacto'].str.contains('.com')).sum() /len(REDES_SOCIALES)) *100
##########################################################################################


#%%
#####################################PUNTO F(GQM)#########################################
proporcion_sede_inactivas = sum(lista_sedes['estado'] == 'Inactivo')/len(lista_sedes) * 100
##########################################################################################


'''
SECCION SEDES: Genera un dataframe a partir de lista_sedes conteniendo los datos esquematizados en el DER
para la entidad SEDES
1. Nos quedamos con las sedes Activas
2. Columnas: ['sede id', 'iso3']
'''
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
consulta = """
               SELECT p.iso3, COUNT(s.sede_id) AS cantidad_sedes,
               FROM PAISES AS p
               LEFT OUTER JOIN SEDES AS s ON p.iso3 = s.iso3
               GROUP BY p.iso3;
              """   
tablaContadorSedes = sql^ consulta

#Agrego resultado anterior a la tabla
consulta = """
               SELECT DISTINCT p.nombre, t.cantidad_sedes, p.iso3
               FROM PAISES AS p
                JOIN tablaContadorSedes AS t
               ON p.iso3 = t.iso3
              """
tablaPaisesYSedes = sql^ consulta

#Junto tabla secciones y tabla sedes
consulta = """
             SELECT DISTINCT sec.tipo_seccion, sed.iso3, sed.sede_id,
             FROM SEDES AS sed
             LEFT JOIN SECCIONES AS sec
             ON sed.sede_id = sec.sede_id
            """
tablaSeccionesYSedes = sql^ consulta

#Cuento cantidad de secciones por sede_id (cuento las veces que se repiten sede_id,
#ya que si sede_id se repite 2 veces significa que tiene dos secciones)
consulta = """
            SELECT DISTINCT COUNT(sede_id) AS cantidad_secciones, sede_id, ANY_VALUE(iso3) AS iso3
            FROM tablaSeccionesYSedes 
            GROUP BY sede_id

    """
tablaCantidadSeccionesPorSede = sql^ consulta

consulta = """
    SELECT ROUND(AVG(t.cantidad_secciones),2) AS promedio_secciones, ANY_VALUE(t.sede_id) AS sede_id, t.iso3
    FROM tablaCantidadSeccionesPorSede as t
    GROUP BY iso3
"""
tablaPromedioSeccionesPorSede = sql^ consulta

consulta = """
        SELECT tc.sede_id, tp.cantidad_sedes, tc.promedio_secciones, tp.nombre, tp.iso3
        FROM tablaPaisesYSedes AS tp
        LEFT JOIN  tablaPromedioSeccionesPorSede AS tc
        ON tc.iso3 = tp.iso3
"""
tablaCantidadSedesYSeccionesPorPais = sql^ consulta
#calculo el flujo en 2022 por pais
consulta = """
    SELECT monto AS "IED 2022 (M U$S)", iso3
    FROM FLUJOS_MONETARIOS
    WHERE fecha == '2022'
"""

tablaIED2022 = sql^consulta

#Uno este resultado con los resultados anteriores de cant de sedes por pais y secciones por sede y ordeno 
consulta = """
    SELECT tc.nombre AS pais, tc.cantidad_sedes AS sedes, tc.promedio_secciones AS "secciones promedio", ti."IED 2022 (M U$S)"
    FROM tablaCantidadSedesYSeccionesPorPais as tc
    LEFT JOIN tablaIED2022 as ti
    ON tc.iso3 = ti.iso3
    ORDER BY tc.cantidad_sedes DESC, tc.nombre
"""
tablaResultado1 = sql^ consulta

#%%
'''
II) Reportar agrupando por región geográfica: a) la cantidad de países en que
Argentina tiene al menos una sede y b) el promedio del IED del año 2022 de
esos países (promedio sobre países donde Argentina tiene sedes). Ordenar
de manera descendente por este último campo.
'''
#Contamos la cantidad de sedes por pais
consulta = '''
SELECT iso3,COUNT(*) AS "paises con sedes argentinas"
FROM SEDES
GROUP BY iso3;
'''
paises_con_sedes_argentinas = sql^ consulta


#A cada páis le asigna una región
consulta = '''
SELECT region_geografica, "paises con sedes argentinas"
FROM paises_con_sedes_argentinas
JOIN REGION ON paises_con_sedes_argentinas.iso3 = REGION.iso3;
'''
tabla_region_geografica_cantidad_sedes = sql^ consulta


# Cuenta la cantidad de paises con sedes argentinas por región
consulta ='''
SELECT t.region_geografica AS "Región geográfica", COUNT(t.region_geografica) AS "Países Con Sedes Argentinas"
FROM tabla_region_geografica_cantidad_sedes AS t
GROUP BY t.region_geografica;
'''
cantidad_paises_region = sql^consulta


#se queda con los flujos del 2022 
consulta = '''
SELECT (*)
FROM FLUJOS_MONETARIOS
WHERE fecha = '2022';
'''
flujos_2022 = sql^consulta

#A cada flujo le asigna una region
consulta = '''
SELECT monto, region_geografica
FROM flujos_2022
JOIN REGION ON flujos_2022.iso3 = REGION.iso3;
'''
flujo_region = sql^consulta


#le saca el promedio por region
consulta = '''
SELECT region_geografica,AVG(monto) AS promedio
FROM flujo_region
GROUP BY region_geografica;
'''
flujo_promedio_region = sql^ consulta

#Hace un natural join entre flujo_promedio_region  y cantidad_paises_region
consulta = '''
SELECT t."Región geográfica", t."Países Con Sedes Argentinas", f.promedio AS "Promedio IED 2022 (M U$S)"
FROM cantidad_paises_region AS t
JOIN flujo_promedio_region AS f ON t."Región geográfica" = f.region_geografica
ORDER BY f.promedio DESC;
'''
tablaResultado2 = sql^consulta

#%%
    
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
        SELECT COUNT(DISTINCT tipo_red) AS cantidad_tipos_redes, ANY_VALUE(nombre) AS pais
        FROM tablaPaisesYRedes
        GROUP BY iso3
"""

tablaResultado3 = sql ^ consulta11


#%%
'''
IV) Confeccionar un reporte con la información de redes sociales, donde se
indique para cada caso: el país, la sede, el tipo de red social y url utilizada.
Ordenar de manera ascendente por nombre de país, sede, tipo de red y
finalmente por url.
'''

#A cada sede_id en REDES_SOCIALES le asigna un iso3 a través de un natural join entre SEDES y REDES_SOCIALES
consulta= '''
SELECT s.sede_id, r.contacto,r.tipo_red,s.iso3
FROM REDES_SOCIALES AS r
JOIN SEDES AS s ON s.sede_id = r.sede_id;
'''
primer_join = sql^consulta


#A cada iso3 le asigna un nombre de pais y ordena en forma ascendente por:
#1.nombre, 2.sede_id , 3. tipo_red y 4.contacto
consulta = '''
SELECT p.nombre AS pais, p1.sede_id AS Sede, p1.contacto AS URL , p1.tipo_red AS "red social"
FROM primer_join AS p1
JOIN PAISES AS p ON p.iso3 = p1.iso3
ORDER BY p.nombre ASC, p1.sede_id ASC,p1.tipo_red ASC, p1.contacto ASC;
'''

tablaResultado4 = sql^consulta

#%%
###################################  GRAFICOS   ############################
'''
i) 
Cantidad de sedes por región geográfica. Mostrarlos ordenados de manera decreciente por dicha cantidad.
'''

tabla_sedes = tabla_region_geografica_cantidad_sedes.groupby('region_geografica').sum().reset_index()
    

unique_regions = tabla_sedes['region_geografica'].unique()
palette = sns.color_palette('coolwarm', len(unique_regions))
color_mapping = dict(zip(unique_regions, palette))

tabla_sedes_sorted = tabla_sedes.sort_values('paises con sedes argentinas', ascending=False)

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

ax = sns.barplot(x='region_geografica', y='paises con sedes argentinas', data=tabla_sedes_sorted, palette=color_mapping) # Gráfico de barras con Seaborn



for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

ax.set_xlabel('Región', fontsize=14, fontweight='bold')
ax.set_ylabel('Cantidad de Sedes', fontsize=14, fontweight='bold')
ax.set_title('Cantidad de Sedes vs Regiones', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)  
plt.tight_layout()
plt.show()
#%%
'''
ii)
Boxplot, por cada región geográfica, del valor correspondiente al promedio del IED 
(para calcular el promedio tomar los valores anuales correspondientes al período 2018-2022) de los países donde Argentina tiene una delegación. 
Mostrar todos los boxplots en una misma figura, ordenados por la mediana de cada región.

'''
promedio_2018_2022 = FLUJOS_MONETARIOS.groupby('iso3')['monto'].mean().reset_index() # Calcular el promedio del IED por país para el período 2018-2022

promedio_region = pd.merge(promedio_2018_2022, REGION, on='iso3') # Natural join promedio_2018_2022 con REGION

promedio_region = promedio_region[promedio_region['monto'] > 0] # Filtrar valores negativos y ceros

ordenado = promedio_region.groupby('region_geografica')['monto'].median().sort_values(ascending=False).index

sns.set(style="whitegrid")


plt.figure(figsize=(14, 8))


ax = sns.boxplot(x='region_geografica', y='monto', data=promedio_region, order=ordenado, palette=color_mapping, showfliers=True)
ax.set_yscale('log')  # escala logarítmica para mejorar la visualización. NOTA: los valores negativos no aparecerán en el gráfico

ticks = [10**i for i in range(-5, 7)]  # from 10^-5 to 10^6
ax.set_yticks(ticks)


ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))

ax.set_xlabel('Región Geográfica', fontsize=12, fontweight='bold')
ax.set_ylabel('Flujo Promedio (2018-2022) [log scale]', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Flujos Promedio por Región Geográfica', fontsize=14, fontweight='bold')

plt.xticks(rotation=45, ha='right', fontsize=12)
sns.despine(trim=True, left=True)
plt.ylim((1e1, 1e6))
plt.tight_layout()
plt.show()
#%%
'''
iii) 
Relación entre el IED de cada país (año 2022 y para todos los países que se tiene información) 
y la cantidad de sedes en el exterior que tiene Argentina en esos países.
'''
tabla_curada = tablaResultado1[~(tablaResultado1['IED 2022 (M U$S)'].isna())]

for i in [3,5,8,9,11]:
    '''
    Se tira los paises que tienen solo un punto en el gráfico
    '''
    tabla_curada = tabla_curada[~(tabla_curada['sedes'] == i)]

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))

# Crear el boxplot
ax = sns.boxplot(x='sedes', y='IED 2022 (M U$S)', data=tabla_curada, palette='viridis')

ax.set_yscale('log') # escala logarítmica para mejorar la visualización. NOTA: los valores negativos no aparecerán en el gráfico

ax.set_xlabel('Cantidad de sedes', fontsize=12, fontweight='bold')
ax.set_ylabel('IED 2022 (M U$S) [log scale]', fontsize=12, fontweight='bold')
ax.set_title('Relacion entre el IED de cada pais y la cantidad de sedes', fontsize=14, fontweight='bold')

plt.xticks(rotation=0, ha='right')

sns.despine(trim=True, left=True)

plt.tight_layout()
plt.show()


#%%
# path = r'C:\Users\Luis Quispe\Desktop\TP01-MLJ\TablasLimpias'
# os.chdir(path)

# SEDES.to_csv('sedes.csv', index=False)
# REDES_SOCIALES.to_csv('redes_sociales.csv', index=False)
# FLUJOS_MONETARIOS.to_csv('flujos_monetarios.csv', index=False)
# SECCIONES.to_csv('secciones.csv', index=False)
# PAISES.to_csv('paises.csv', index=False)
# REGION.to_csv('region.csv', index=False)

# tablaResultado1.to_csv(r'.\Anexo\consulta_1.csv',index = False)
# tablaResultado2.to_csv(r'.\Anexo\consulta_2.csv',index = False)
# tablaResultado3.to_csv(r'.\Anexo\consulta_3.csv',index = False)
# tablaResultado4.to_csv(r'.\Anexo\consulta_4.csv',index = False)

# first_rows = tablaResultado1.head(13)
# first_rows.to_csv('first_rows.csv', index=False)