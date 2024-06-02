import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import tree
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import graphviz
from sklearn import metrics
import random
#%%
#Esto te abre los gráficos en una ventana aparte (CORRERLO ES OPCIONAL)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

path= r'C:\Users\Luis Quispe\Desktop\Labo_Datos\TP02\Archivos TP-02-20240527'
os.chdir(path)

#%%
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
#%%
#esta parte rota y flipea todas las letras en el data frame
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values

transformed_images = np.array([flip_rotate(image) for image in images])
flattened_images = transformed_images.reshape(transformed_images.shape[0], -1)
data1 = pd.DataFrame(np.column_stack((labels, flattened_images)))
#%%
#cuento cantidad de letras
cantidad_letras = data1[0].value_counts()
# indices = data1[0][data1[0] == 'M']
#%%
matriz_valores = np.array(data1.drop(0, axis=1).astype(float))


vec_promedio = np.mean(matriz_valores, axis=0)

image_vec_promedio = vec_promedio.reshape(28, 28)

plt.imshow(image_vec_promedio)
plt.title("Average Image")
plt.show()
#%%
# Encontrar las columnas con valores promedio entre 0 y 1
columnas_menores_1 = np.where((vec_promedio >= 0) & (vec_promedio <= 1))[0]

# Transformar el vector promedio en una imagen
imagen_vec_promedio = vec_promedio.reshape(28, 28)

# Visualizar la imagen promedio con recuadros en las columnas de bajo valor
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
#PTO 1B (LETRAS)
letra_promedio = pd.DataFrame(flattened_images.astype(float))
letra_promedio.insert(0, 'label', labels)
grouped_data = letra_promedio.groupby('label').mean()


grid_size = int(np.ceil(np.sqrt(grouped_data.shape[0])))
fig, axes = plt.subplots(2, int(np.ceil(grouped_data.shape[0] / 2)), figsize=(15, 8))
axes = axes.flatten()


for i, (label, mean_image) in enumerate(grouped_data.iterrows()):
    ax = axes[i]
    mean_image = mean_image.values.reshape(28, 28)
    ax.imshow(mean_image)
    ax.set_title(f'Letra {label}')
    ax.axis('off')

# Turn off any unused subplots
for i in range(grouped_data.shape[0], len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#%%
#PTO 1B (DISTANCIA)
#ME DA EL HEATMAP CON LA DISTANCIA ENTRE LOS PARES DE LETRAS QUE QUIERO NORMALIZADAS

selected_letters = ['L', 'I', 'M', 'N', 'E', 'F', 'O', 'Q', 'X', 'V', 'B', 'D', 'C', 'U', 'Y']
average_images_selected = grouped_data.loc[selected_letters]
distances = cdist(average_images_selected, average_images_selected, metric='euclidean')

min_distance = np.min(distances)
max_distance = np.max(distances)
distances_normalized = (distances - min_distance) / (max_distance - min_distance)

plt.figure(figsize=(10, 8))
sns.heatmap(distances_normalized, xticklabels=selected_letters, yticklabels=selected_letters, cmap='viridis', annot=True, fmt=".2f")
plt.title('Mapa de calor de las distancias normalizadas entre imágenes promedio de letras seleccionadas')
plt.xlabel('Letra')
plt.ylabel('Letra')
plt.show()

#%%
#NUEVO PTO 1C
from scipy.spatial.distance import euclidean

for i in range(len(data)):
    row = data.iloc[i].drop(0).values
    letra = data.iloc[i][0]
    image_array = np.array(row).astype(np.float32)
    transformed_image = flip_rotate(image_array)
    images.append(transformed_image)
    labels.append(letra)

images = np.array(images)
labels = np.array(labels)

# Letras seleccionadas para el análisis
selected_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
average_images = {}
avg_distances = {}

# Calcular el promedio de las imágenes y la desviación estándar para cada letra seleccionada
for letter in selected_letters:
    images_letter = images[labels == letter]
    avg_image = np.mean(images_letter, axis=0)
    average_images[letter] = avg_image
    
    # Calcular las distancias de cada imagen de la letra al promedio
    distances = [euclidean(img.flatten(), avg_image.flatten()) for img in images_letter]
    avg_distance = np.mean(distances)
    avg_distances[letter] = avg_distance

# Configurar el gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_distances.keys(), avg_distances.values(), color='skyblue')

# Añadir los valores de desviación estándar encima de cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')
    
# Añadir una línea vertical en la distancia promedio especificada
plt.axhline(y=1799, color='r', linestyle='--', label='Promedio = 1799')
plt.legend()
plt.xlabel('Letras')
plt.ylabel('Distancia Promedio')
plt.title('Distancia Promedio de cada tipo de letra a su promedio')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#%%
#VIEJO C
# Dictionary to store standard deviations for each letter
std_dev_per_letter = {}

# Loop through each letter
for letter in data1[0].unique():
    # Get vectors for the current letter
    vectors_letter = data1[data1[0] == letter].drop(0, axis=1).astype(float)
    
    # Calculate distances for the current letter
    distances_letter = cdist(vectors_letter, vectors_letter, metric='euclidean')
    
    # Calculate standard deviation for the distances
    std_dev = np.std(distances_letter)
    
    # Store standard deviation in the dictionary
    std_dev_per_letter[letter] = std_dev

# Create lists to store letters and their corresponding standard deviations
letters = list(std_dev_per_letter.keys())
std_devs = list(std_dev_per_letter.values())

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(letters, std_devs, color='skyblue')
plt.xlabel('Letter')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Distances for Each Letter')
plt.xticks(rotation=45)
plt.show()

#%%
#PUNTO 2
data1_frame_la = data1[(data1[0] == 'L') | (data1[0] == 'A')]
cantidad_por_letra = data1_frame_la[0].value_counts()
X_dev, X_eval, y_dev, y_eval = train_test_split(data1_frame_la.drop(0, axis =1),data1_frame_la[0],random_state=1,test_size=0.2)

#%% PUNTO 2_D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Voy a buscar los indices donde hay un maximo en alguna clase para A y para L
data1_L = data1_frame_la[data1_frame_la[0] == 'L']
data1_A = data1_frame_la[data1_frame_la[0] == 'A']
maximo_L = data1_L.drop(data1_L.columns[0], axis = 1).max().max()
maximo_A = data1_A.drop(data1_A.columns[0], axis = 1).max().max()

print(f"El maximo pixel entre las clases de la letra L es: {maximo_L}")
print(f"El maximo pixel entre las clases de la letra A es: {maximo_A}")

indices_A_maximo = data1_frame_la[(data1_frame_la[data1_frame_la.columns[0]] == 'A') & (data1_frame_la == 255).any(axis=1)].index
print(f"El maximo pixel entre las clases de la letra A se encuentra en los indices: {indices_A_maximo}")

indices_L_maximo = data1_frame_la[(data1_frame_la[data1_frame_la.columns[0]] == 'L') & (data1_frame_la == 255).any(axis=1)].index
print(f"El maximo pixel entre las clases de la letra L se encuentra en los indices: {indices_L_maximo}")


# Seleccionamos 3 atributos mas cercanos el maximo pixel para A(en el indice 69 hay un max)

X_train_caso1 = X_dev.iloc[:, 68:70]
X_test_caso1 = X_eval.iloc[:, 68:70]
# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso1, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso1 = model.predict(X_test_caso1) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso1)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.53125

#Seleccionamos 3 atributos mas cercanos el maximo pixel para L (en el indice 708 hay un max)
X_train_caso2 = X_dev.iloc[:, 707:709]
X_test_caso2 = X_eval.iloc[:, 707:709]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso2, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso2 = model.predict(X_test_caso2) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso2)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.5354166666666667

#Voy a probar tomando el maximo pero del promedio entre las L y las A juntas
promedio_LA = data1_frame_la.drop(columns=[0]).mean()
#Encuetro el indice que contiene al maximo
indice_maximo = promedio_LA.idxmax()
print(f"El pixel de mayor promedio se encuentra en la posicion: {indice_maximo}")
#El pixel de mayor promedio se encuentra en la posicion: 406

#Voy a entrenar tomando los 3 atributos mas cercanos al maximo global
X_train_caso3 = X_dev.iloc[:, 405:407]
X_test_caso3 = X_eval.iloc[:, 405:407]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso3, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso3 = model.predict(X_test_caso3) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso3)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.75

""" Como el modelo con mayor accuracy fue el tercero(tomando 3 atributos cercanos al maximo del promedio de L y A), 
voy a usar  este maximo pero tomando primero los 7 mas cercanos, los 10 mas cercanos y los 20 mas cercanos
(el maximo se encuentra en el indice 406) """

# Seleccionamos 7 atributos mas cercanos el maximo pixel para el promedio
X_train_caso4 = X_dev.iloc[:, 402:409]
X_test_caso4 = X_eval.iloc[:, 402:409]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso4, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso4 = model.predict(X_test_caso4) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso4)

print(f"Accuracy con 7 atributos: {score}")
#Accuracy con 7 atributos: 0.88125

# Seleccionamos 10 atributos mas cercanos el maximo pixel para el promedio
X_train_caso5 = X_dev.iloc[:, 401:411]
X_test_caso5 = X_eval.iloc[:, 401:411]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso5, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso5 = model.predict(X_test_caso5) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso5)

print(f"Accuracy con 10 atributos: {score}")
#Accuracy con 10 atributos: 0.9270833333333334

# Seleccionamos 20 atributos mas cercanos el maximo pixel para el promedio
X_train_caso6 = X_dev.iloc[:, 396:416]
X_test_caso6 = X_eval.iloc[:, 396:416]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso6 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos: {score}")
#Accuracy con 20 atributos: 0.9885416666666667

#%% 2_e
"""
Voy a comparar el ultimo modelo tomando vecinos {k= 5, 10, 15, 20, 50} a ver que sucede
"""

X_train_caso6 = X_dev.iloc[:, 396:416]
X_test_caso6 = X_eval.iloc[:, 396:416]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 2) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos y 2 vecinos: {score}")
#Accuracy con 20 atributos y 2 vecinos: 0.9864583333333333


model = KNeighborsClassifier(n_neighbors = 5) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos y 5 vecinos: {score}")
#Accuracy con 20 atributos y 5 vecinos en el KNN: 0.9885416666666667


model = KNeighborsClassifier(n_neighbors = 10) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso6 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos: y 10 vecinos en el KNN {score}")
#Accuracy con 20 atributos: y 10 vecinos en el KNN 0.9864583333333333



model = KNeighborsClassifier(n_neighbors = 15) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso6 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos y 15 vecinos en el KNN: {score}")
#Accuracy con 20 atributos y 15 vecinos en el KNN: 0.9864583333333333


model = KNeighborsClassifier(n_neighbors = 20) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso6 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos y 20 vecinos en el KNN: {score}")
#Accuracy con 20 atributos y 20 vecinos en el KNN: 0.984375


model = KNeighborsClassifier(n_neighbors = 50) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso6 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)

print(f"Accuracy con 20 atributos y 50 vecinos en el KNN: {score}")
#Accuracy con 20 atributos y 50 vecinos en el KNN: 0.96875

###Ahora vamos a tomar los 30 atributos del centro de mi dataframe test (indice 392) e ir variando los k

X_train_caso7 = X_dev.iloc[:, 377:407]
X_test_caso7 = X_eval.iloc[:, 377:407]

# Ajustamos el modelo KNN
model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso7, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7= model.predict(X_test_caso7) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso7)

print(f"Accuracy con 30 atributos y 3 vecinos en el KNN: {score}")
#Accuracy con 30 atributos y 3 vecinos en el KNN: 0.9864583333333333

model = KNeighborsClassifier(n_neighbors = 5) # modelo en abstracto
model.fit(X_train_caso7, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7= model.predict(X_test_caso7) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso7)

print(f"Accuracy con 30 atributos y 5 vecinos en el KNN: {score}")
#Accuracy con 30 atributos y 5 vecinos en el KNN: 0.9854166666666667

model = KNeighborsClassifier(n_neighbors = 10) # modelo en abstracto
model.fit(X_train_caso7, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7= model.predict(X_test_caso7) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso7)

print(f"Accuracy con 30 atributos y 10 vecinos en el KNN: {score}")
#Accuracy con 30 atributos y 10 vecinos en el KNN: 0.9833333333333333

model = KNeighborsClassifier(n_neighbors = 20) # modelo en abstracto
model.fit(X_train_caso7, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7= model.predict(X_test_caso7) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso7)

print(f"Accuracy con 30 atributos y 20 vecinos en el KNN: {score}")
#Accuracy con 30 atributos y 20 vecinos en el KNN: 0.9791666666666666

model = KNeighborsClassifier(n_neighbors = 50) # modelo en abstracto
model.fit(X_train_caso7, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7= model.predict(X_test_caso7) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso7)

print(f"Accuracy con 30 atributos y 50 vecinos en el KNN: {score}")
#Accuracy con 30 atributos y 50 vecinos en el KNN: 0.9635416666666666

#%%
#PUNTO 3 
data1_frame_vocales = data1[ (data1[0] == 'A') | (data1[0] == 'E') | (data1[0] == 'I') | (data1[0] == 'O') | (data1[0] == 'U')]

X_dev1, X_eval1, y_dev1, y_eval1 = train_test_split(data1_frame_vocales.drop(0, axis =1),data1_frame_vocales[0],random_state=1,test_size=0.2)

def crear_arbol(profundidad, X, Y):
    clf_info = tree.DecisionTreeClassifier(max_depth= profundidad)
    clf_info = clf_info.fit(X, Y)
    return clf_info

# Range of depths to try
max_depths = range(1, 11)
best_depth = 0
best_accuracy = 0
best_tree = None

# Train and evaluate trees with different depths
accuracies = []
for depth in max_depths:
    clf = crear_arbol(depth, X_dev1, y_dev1)
    y_pred = clf.predict(X_eval1)
    accuracy = accuracy_score(y_eval1, y_pred)
    accuracies.append(accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
        best_tree = clf

print(f'Best Depth: {best_depth}, Best Accuracy: {best_accuracy}')
#%%
# Visualize the best decision tree
feature_names = X_dev1.columns.tolist()
class_names = y_dev1.unique().tolist()
plt.figure(figsize=(20,10))
tree.plot_tree(best_tree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=1)
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,10))
# Plot the accuracies for each depth
plt.plot(max_depths, accuracies, marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.show()

















