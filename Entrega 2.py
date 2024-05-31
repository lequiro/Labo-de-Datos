import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
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
#PTO 1C
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

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_dev, y_dev) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_eval) # me fijo qué clases les asigna el modelo a mis datos
metrics.accuracy_score(y_eval, Y_pred)

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


# Seleccionamos 3 atributos mas cercanos el maximo pixel para A(en el indice 170 hay un max)

X_train_caso1 = X_dev.iloc[:, 169:171]
X_test_caso1 = X_eval.iloc[:, 169:171]
# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso1, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso1 = model.predict(X_test_caso1) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso1)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.5052083333333334

#Seleccionamos 3 atributos mas cercanos el maximo pixel para L (en el indice 619 hay un max)
X_train_caso2 = X_dev.iloc[:, 618:620]
X_test_caso2 = X_eval.iloc[:, 618:620]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso2, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso2 = model.predict(X_test_caso2) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso2)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.6958333333333333

#Voy a probar tomando el maximo pero del promedio entre las L y las A juntas
promedio_LA = data1_frame_la.drop(columns=[0]).mean()
#Encuetro el indice que contiene al maximo
indice_maximo = promedio_LA.idxmax()
print(f"El pixel de mayor promedio se encuentra en la posicion: {indice_maximo}")

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

#%%
#PUNTO 3 
data1_frame_vocales = data1[ (data1[0] == 'A') | (data1[0] == 'E') | (data1[0] == 'I') | (data1[0] == 'O') | (data1[0] == 'U')]
X_dev1, X_eval1, y_dev1, y_eval1 = train_test_split(data1_frame_vocales.drop(0, axis =1),data1_frame_vocales[0],random_state=1,test_size=0.2)
