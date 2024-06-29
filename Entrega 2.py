"""
Trabajo Práctico 02
Materia: Laboratorio de datos - FCEyN - UBA
Integrantes: Otermín Juana, Quispe Rojas Luis Enrique , Vilcovsky Maia

Este código se ocupa principalmente de realizar análisis de imágenes de letras utilizando
técnicas de manipulación de datos, análisis estadístico, visualización y algoritmos de 
machine learning. El contenido del archivo "emnist_letters_tp.csv" contiene imágenes de 
letras, donde cada fila representa una imagen de 28x28 píxeles y una etiqueta 
correspondiente a la letra representada.

Fecha  : 2024-06-04
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree,export_text
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import euclidean
#%%
#Esto te abre los gráficos en una ventana aparte (CORRERLO ES OPCIONAL)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

path = input("Coloque la ruta de acceso a su archivo csv: ")
os.chdir(path)
#%%
#importa el dataframe del conjunto de datos MNIST
data = pd.read_csv("emnist_letters_tp.csv", header= None)
#%%
#FUNCIONES
def flip_rotate(image):
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

def crear_arbol(profundidad, X, Y):
    clf_info = DecisionTreeClassifier(max_depth=profundidad)
    clf_info = clf_info.fit(X, Y)
    return clf_info

def desviacion_promedio(letra):
    letra_seleccionada = np.array(data1[data1[0] == letra].loc[:,1:],dtype=float)
    letra_seleccionada_promedio= np.mean(letra_seleccionada, axis=0)
    difference = letra_seleccionada - letra_seleccionada_promedio
    estandar_vec = np.std(difference,axis=1)
    estandar_vec_mean = estandar_vec.mean()
    return estandar_vec_mean

#%%
#esta parte rota y flipea todas las letras en el data frame
labels = data.iloc[:, 0].values #se queda con las letras correspondientes a cada fila
images = data.iloc[:, 1:].values #se queda con el resto del dataframe correspondientes a los valores por pixel

transformed_images = np.array([flip_rotate(image) for image in images]) #aplica a cada fila la función flip_rotate
flattened_images = transformed_images.reshape(transformed_images.shape[0], -1)
data1 = pd.DataFrame(np.column_stack((labels, flattened_images)))

#cuento cantidad de letras
cantidad_letras = data1[0].value_counts() #cuenta la cantidad de imagenes por letra
#%%
#PTO 1A
#Calcula la imagen promedio de todas las letras
matriz_valores = np.array(data1.drop(0, axis=1).astype(float))
vec_promedio = np.mean(matriz_valores, axis=0)
image_vec_promedio = vec_promedio.reshape(28, 28)

plt.figure()
plt.imshow(image_vec_promedio)
plt.title("Promedio de todas las letras")
plt.show()

# Encontrar las columnas con valores promedio entre 0 y 1
columnas_menores_1 = np.where((vec_promedio >= 0) & (vec_promedio <= 1))[0]

# Transformar el vector promedio en una imagen
imagen_vec_promedio = vec_promedio.reshape(28, 28)

# Visualizar la imagen promedio con recuadros en las columnas de bajo valor
plt.figure()
plt.imshow(imagen_vec_promedio, cmap='viridis')
plt.title("Promedio de todas las letras")

# Agregar recuadros alrededor de las columnas de bajo valor
for col in columnas_menores_1:
    x = col % 28
    y = col // 28
    plt.gca().add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, edgecolor='red', facecolor='none', lw=2))

# Mostrar la imagen
plt.show()

#%%
#PTO 1B (PRIMER ANALISIS)
#Calcula la imagen promedio por letra

letra_promedio = pd.DataFrame(flattened_images.astype(float))
letra_promedio.insert(0, 'label', labels)
group_por_letra = letra_promedio.groupby('label').mean()


grid_size = int(np.ceil(np.sqrt(group_por_letra.shape[0])))
fig, axes = plt.subplots(2, int(np.ceil(group_por_letra.shape[0] / 2)), figsize=(15, 8))
axes = axes.flatten()


for i, (label, mean_image) in enumerate(group_por_letra.iterrows()):
    ax = axes[i]
    mean_image = mean_image.values.reshape(28, 28)
    ax.imshow(mean_image)
    ax.set_title(f'Letra {label}')
    ax.axis('off')

for i in range(group_por_letra.shape[0], len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#%%
#PTO 1B (SEGUNDO ANALISIS)
#ME DA EL HEATMAP CON LA DISTANCIA ENTRE LOS PARES DE LETRAS QUE QUIERO NORMALIZADAS

letras_seleccionadas = ['L', 'I', 'M', 'N', 'E', 'F', 'O', 'Q', 'X', 'V', 'B', 'D', 'C', 'U', 'Y']
average_images_selected = group_por_letra.loc[letras_seleccionadas]
distances = cdist(average_images_selected, average_images_selected, metric='euclidean')

min_distance = np.min(distances)
max_distance = np.max(distances)
distances_normalized = (distances - min_distance) / (max_distance - min_distance)

plt.figure(figsize=(17.28, 8.1))
sns.heatmap(distances_normalized, xticklabels=letras_seleccionadas, yticklabels=letras_seleccionadas, cmap='viridis', annot=True, fmt=".2f")
plt.title('Mapa de calor de las distancias normalizadas entre imágenes promedio de letras seleccionadas')
plt.xlabel('Letra')
plt.ylabel('Letra')
plt.show()


#%%
#PTO 1C
#Calculo de la desvicación estándar a la imagen promedio
# Letras seleccionadas para el análisis
letras_seleccionadas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
diccionario_total = {}

for i in letras_seleccionadas:
    valor_promedio = desviacion_promedio(i)
    diccionario_total.update({i: valor_promedio})
    
# Graficar el diagrama de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(diccionario_total.keys(), diccionario_total.values(), color='skyblue')
plt.xlabel('Letra')
plt.ylabel('Desvicación estándar promedio')
plt.title('Desvicación estándar promedio para cada letra')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mayor claridad

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  
                 textcoords="offset points",
                 ha='center', va='bottom')

# Agregar una línea horizontal en el valor de la letra 'C'
plt.axhline(y=diccionario_total['C'], color='r', linestyle='--', label='C')
plt.legend()
#%%
#PUNTO 2
data1_frame_la = data1[(data1[0] == 'L') | (data1[0] == 'A')]
cantidad_por_letra = data1_frame_la[0].value_counts()
X_dev, X_eval, y_dev, y_eval = train_test_split(data1_frame_la.drop(0, axis =1),data1_frame_la[0],random_state=1,test_size=0.2)

#%% PUNTO 2_D 
"""
Tomamos de criterio para elegir 3 atributos del dataframe que estos sean los que tengan
la maxima distancia entre las letras A y L (promediando sus clases) para poder tener
mayor dispersion de los datos
"""
#Busca el promedio de las clases para la letra L y la letra A
promedios_A = np.mean(data1_frame_la[data1_frame_la.loc[:,0] == "A"].loc[:,1:], axis=0)
promedios_L = np.mean(data1_frame_la[data1_frame_la.loc[:,0] == "L"].loc[:,1:], axis =0)

#Busca la maxima distancia entre ellas
distancia_max = np.max(np.abs(promedios_L-promedios_A))
print(distancia_max)
# 188.66833333333335 es la distancia maxima entre L y A

#Busca el indice del atributo donde se encuentra la maxima distancia entre ellas
indice_distancia = np.argmax(np.abs(promedios_L - promedios_A))

print(indice_distancia)
#439 es la columna donde se da esta distancia maxima

"""
Creamos un boxplot para visualizar como se distribuyen los valores para A y para L
en esta columna con distancia maxima
"""
valores_L = data1_frame_la[data1_frame_la.loc[:,0] == "L"].iloc[:,indice_distancia]
valores_A = data1_frame_la[data1_frame_la.loc[:,0] == "A"].iloc[:,indice_distancia]

#Creamos boxplot
plt.figure(figsize=(17.28, 8.1))
plt.boxplot([valores_L, valores_A], labels = ["L", "A"])
plt.title(f'Boxplot de la columna {indice_distancia} con distancia máxima')
plt.xlabel("Categoria")
plt.ylabel("Valores")
plt.show()

"""
Para seguir tomando otros 3 atributos voy a armar una lista con las maximas distacias ]
entre las clases L y A con un rango de distancia entre 150 y 188
"""
x = np.abs(promedios_L - promedios_A)
buscador = np.argwhere((x >= 150) & (x <= 188)).flatten()

print(buscador)

"""
las maximas distancias se encuentran en las columnas [382,383,384,402,409,410,411,
412,413,429,437,438,439(maximo),440,465,466,467,468,494,495]
"""
#tomo 3 atributos donde en uno se encuentra la maxima distancia
X_train_caso1 = X_dev.iloc[:, [495,439,412]]
X_test_caso1 = X_eval.iloc[:, [495,439,412]]

model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train_caso1, y_dev) 
Y_pred_caso1 = model.predict(X_test_caso1) 
score = accuracy_score(y_eval, Y_pred_caso1)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.96875

#Elijo otros 3 atributos pero consecutivos
X_train_caso2 = X_dev.iloc[:, 382:384]
X_test_caso2 = X_eval.iloc[:, 382:384]


model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train_caso2, y_dev) 
Y_pred_caso2 = model.predict(X_test_caso2) 
score = accuracy_score(y_eval, Y_pred_caso2)
print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.9302083333333333

X_train_caso3 = X_dev.iloc[:, 465:467]
X_test_caso3 = X_eval.iloc[:, 465:467]


#Elijo otros 3 consecutivos
model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso3, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso3 = model.predict(X_test_caso3) # me fijo qué clases les asigna el modelo a mis datos
score = accuracy_score(y_eval, Y_pred_caso3)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.9291666666666667

#Otro con atributos random dentro de la lista 
X_train_caso4 = X_dev.iloc[:, [402,429,495]]
X_test_caso4 = X_eval.iloc[:, [402,429,495]]



model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso4, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso4 = model.predict(X_test_caso4) # me fijo qué clases les asigna el modelo a mis datos
score = accuracy_score(y_eval, Y_pred_caso4)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.903125

"""
Ahora voy a tomar probar tomando mas cantidad de atributos, primero voy a tomar 5, luego 10 y finalmente
voy a tomar 15
"""

#Modelo con 5 atributos tratando de que no sean consecutivos
X_train_caso5 = X_dev.iloc[:, [382,402,411,437,495]]
X_test_caso5 = X_eval.iloc[:, [382,402,411,437,495]]


model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train_caso5, y_dev) 
Y_pred_caso5 = model.predict(X_test_caso5) 
score = accuracy_score(y_eval, Y_pred_caso5)
print(f"Accuracy con 5 atributos: {score}")
#Accuracy con 5 atributos: 0.971875

#Modelo con 10 atributos tratando de que no sean consecutivos
X_train_caso6 = X_dev.iloc[:, [382,402,409,411,429,437,440,465,468,494]]
X_test_caso6 = X_eval.iloc[:, [382,402,409,411,429,437,440,465,468,494]]

model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train_caso6, y_dev) 
Y_pred_caso6 = model.predict(X_test_caso6)
score = accuracy_score(y_eval, Y_pred_caso6)
print(f"Accuracy con 10 atributos: {score}")
#Accuracy con 10 atributos: 0.9760416666666667


#Modelo con 15 atributos incluyendo  al de distancia maxima
X_train_caso7 = X_dev.iloc[:, [382,383,384,402,409,411,413,429,437,439,465,467,468,494,495]]
X_test_caso7 = X_eval.iloc[:, [382,383,384,402,409,411,413,429,437,439,465,467,468,494,495]]


model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train_caso7, y_dev) 
Y_pred_caso7 = model.predict(X_test_caso7) 
score = accuracy_score(y_eval, Y_pred_caso7)
print(f"Accuracy con 15 atributos: {score}")
#Accuracy con 15 atributos: 0.9854166666666667
#%%
#Esto dibuja los pixeles seleccionados en el primer modelo para las letras 'L' y 'A'
array_L = np.array(promedios_L,dtype=float).reshape(28,28)
array_A = np.array(promedios_A,dtype=float).reshape(28,28)
# Definir los índices para resaltar
indices = [402, 429, 495]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Gráfico para la letra 'L'
array_resaltado_L = np.zeros((28, 28))
for idx in indices:
    x, y = divmod(idx, 28)  
    array_resaltado_L[x, y] = 1

axs[0].imshow(array_L)
axs[0].imshow(array_resaltado_L, cmap='pink', alpha=0.3)
axs[0].set_title('Letra L')

# Gráfico para la letra 'A'
array_resaltado_A = np.zeros((28, 28))
for idx in indices:
    x, y = divmod(idx, 28)  
    array_resaltado_A[x, y] = 1 

axs[1].imshow(array_A)
axs[1].imshow(array_resaltado_A, cmap='pink', alpha=0.3)
axs[1].set_title('Letra A')
plt.show()
#%% Punto 2-E 
"""
Voy a comparar el ultimo modelo que tiene 15 atributos de distancia maxima en promedio entre L y A
tomando vecinos {k= 2, 5, 10, 15, 20, 50} a ver que sucede
"""

k_values = [2, 5, 10, 15, 20, 50]
accuracy_train7 = []
accuracy_test7 = []

X_train_caso7 = X_dev.iloc[:, [382,383,384,402,409,411,413,429,437,439,465,467,468,494,495]]
X_test_caso7 = X_eval.iloc[:, [382,383,384,402,409,411,413,429,437,439,465,467,468,494,495]]

# Iterar sobre los valores de K
for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_caso7, y_dev)
    
    # Calcula el accuracy para los datos de entrenamiento
    Y_pred_caso7 = model.predict(X_train_caso7)
    train_accuracy = accuracy_score(y_dev, Y_pred_caso7)
    accuracy_train7.append(train_accuracy)
    
    # Calcula el accuracy para los datos de test
    Y_pred_caso7 = model.predict(X_test_caso7)
    test_accuracy7 = accuracy_score(y_eval, Y_pred_caso7)
    accuracy_test7.append(test_accuracy7)

#Graficamos accuracy vs valores de K
plt.figure(figsize=(10, 6))


plt.plot(k_values, accuracy_train7, label='Accuracy del entrenamiento', marker='o')
plt.plot(k_values, accuracy_test7, label='Accuracy del test', marker='o')


plt.xlabel('Valores de K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Valores de K')
plt.legend()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xticks(range(0, max(k_values)+1, 5)) 

plt.show()

# Mostrar los valores de accuracy para train y test
print(f"Accuracy de entrenamiento con 15 atributos con k=2: {accuracy_train7[0]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=5: {accuracy_train7[1]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=10: {accuracy_train7[2]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=15: {accuracy_train7[3]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=10: {accuracy_train7[4]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=50: {accuracy_train7[5]}")
print(f"Accuracy de test con 15 atributos con k=2: {accuracy_test7[0]}")
print(f"Accuracy de test con 15 atributos con k=5: {accuracy_test7[1]}")
print(f"Accuracy de test con 15 atributos con k=10: {accuracy_test7[2]}")
print(f"Accuracy de test con 15 atributos con k=15: {accuracy_test7[3]}")
print(f"Accuracy de test con 15 atributos con k=10: {accuracy_test7[4]}")
print(f"Accuracy de test con 15 atributos con k=50: {accuracy_test7[5]}")

#%%
#PUNTO 3A
#Genera el nuevo data frame con las vocales.
#Separamos en train y test(heldout)
data1_frame_vocales = data1[ (data1[0] == 'A') | (data1[0] == 'E') | (data1[0] == 'I') | (data1[0] == 'O') | (data1[0] == 'U')]
X_train1, X_eval1, y_train1, y_eval1 = train_test_split(data1_frame_vocales.drop(0, axis =1),data1_frame_vocales[0],random_state=1,test_size=0.2)

#%%
#PUNTO 3B y 3C
# Ajusto el arbol de decision variando las alturas y los distintis hiperparametros.

param_grid = {
    'max_depth': range(1, 11),
    'criterion': ['gini', 'entropy']
}
#Entreno y evaluo haciendo cross validacion sobre los datos del train
kf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, scoring='accuracy')
kf.fit(X_train1, y_train1)

results = kf.cv_results_

# Tome los mejores parametros
best_params = kf.best_params_
best_score = kf.best_score_

print(f'Mejores parámetros: {best_params}')
print(f'Accuracy : {best_score}')
#Se realiza el grafico para visualizar el codo
plt.figure()
plt.plot(range(1,11), results["mean_test_score"][0:10], marker='o', label = "Gini" )
plt.plot(range(1,11), results["mean_test_score"][10:20], marker='o', label = "Entropia" )
plt.xlabel('Profundidad del árbol')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Profundidad')
plt.legend()
plt.show()

#Genera una visualización del mejor árbol de decisión elijiendo como profundidad 5 y como hiperparametro entropia
best_params = {"criterion": "entropy", "max_depth": 4 }
best_tree = DecisionTreeClassifier(**best_params)
best_tree.fit(X_train1, y_train1)
plt.figure(figsize=(20, 10))
plot_tree(best_tree, feature_names=X_train1.columns, class_names=y_train1.unique().astype(str), filled=True, rounded=True, fontsize=2)
plt.show()
#%% 
#Importo el tree como txt para mejor visualizacion
tree_text = export_text(best_tree, feature_names=list(X_train1.columns))
output_file = "decision_tree.txt"
with open(output_file, "w") as f:
    f.write(tree_text) 
#%%
#Evaluo en el held out 
heldout_predictions = best_tree.predict(X_eval1)
sum(heldout_predictions == y_eval1)

cm = confusion_matrix(y_eval1, heldout_predictions, labels=y_eval1.unique())

# Grafica la Matriz de Confusión sobre el held out
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_eval1.unique())
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión sobre la evaluación en el held out')
plt.show()

accuracy = accuracy_score(y_eval1, heldout_predictions)
precision = precision_score(y_eval1, heldout_predictions, labels=y_eval1.unique(), average='macro')
recall = recall_score(y_eval1, heldout_predictions, labels=y_eval1.unique(), average='macro')
f1 = f1_score(y_eval1, heldout_predictions, labels=y_eval1.unique(), average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


#%%
data1_frame_oi = data1[(data1[0] == 'O') | (data1[0] == 'I')]
promedios_O = np.mean(data1_frame_oi[data1_frame_oi.loc[:,0] == "O"].loc[:,1:], axis=0)
promedios_I = np.mean(data1_frame_oi[data1_frame_oi.loc[:,0] == "I"].loc[:,1:], axis =0)
#Esto dibuja los pixeles seleccionados en el primer modelo para las letras 'L' y 'A'
array_O= np.array(promedios_O,dtype=float).reshape(28,28)
array_I = np.array(promedios_I,dtype=float).reshape(28,28)
# Definir los índices para resaltar
indices_I = [469,628,403,544,678]
indices_O = [469,628,403,544]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Gráfico para la letra 'L'
array_resaltado_O = np.zeros((28, 28))
for idx in indices_O:
    x, y = divmod(idx, 28)  
    array_resaltado_O[x, y] = 1

axs[0].imshow(array_O)
axs[0].imshow(array_resaltado_O, cmap='pink', alpha=0.3)
axs[0].set_title('Letra O')

# Gráfico para la letra 'A'
array_resaltado_I = np.zeros((28, 28))
for idx in indices_I:
    x, y = divmod(idx, 28)  
    array_resaltado_I[x, y] = 1 

axs[1].imshow(array_I)
axs[1].imshow(array_resaltado_I, cmap='pink', alpha=0.3)
axs[1].set_title('Letra I')
plt.show()
