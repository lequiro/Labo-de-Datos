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

#%% PUNTO 2_D : PARA EL MAXIMO VALOR DE DISTANCIA
promedios_L = np.mean(data1_frame_la[data1_frame_la.loc[:,0] == "A"].loc[:,1:], axis=0)
promedios_A = np.mean(data1_frame_la[data1_frame_la.loc[:,0] == "L"].loc[:,1:], axis =0)


distancia = np.argmax(np.abs(promedios_L - promedios_A))
distancia_max = np.max(np.abs(promedios_L-promedios_A))
# 188.66833333333335 es la distancia maxima entre L y A
print(distancia)
#439 es la columna donde se da esta distancia maxima
valores_L = data1_frame_la[data1_frame_la.loc[:,0] == "L"].iloc[:,distancia]
valores_A = data1_frame_la[data1_frame_la.loc[:,0] == "A"].iloc[:,distancia]

#Creamos boxplot
plt.figure(figsize=(10,6))
plt.boxplot([valores_L, valores_A], labels = ["L", "A"])
plt.title(f'Boxplot de la columna {distancia} con distancia máxima')
plt.xlabel("Categoria")
plt.ylabel("Valores")
plt.show()
#%%
"""
Boxplot de los valores de A y de L para las columnas con una distancia entre 150 y 188

"""
x = np.abs(promedios_L - promedios_A)
buscador = np.argwhere((x >= 150) & (x <= 188)).flatten()

print(buscador)
#[[382],[383],[384],[402],[409],[410],[411],[412],[413],[429],[437],[438],[440],[465],[466],[467],[468],[494],[495]]

for valores in buscador:
    valores_L = data1_frame_la[data1_frame_la.loc[:,0] == "L"].iloc[:,valores]
    valores_A = data1_frame_la[data1_frame_la.loc[:,0] == "A"].iloc[:,valores]

    plt.figure(figsize=(10,6))
    plt.boxplot([valores_L, valores_A], labels = ["L", "A"])
    plt.title(f'Boxplot de la columna: {valores}')
    plt.xlabel("Categoria")
    plt.ylabel("Valores")
    plt.show()

#%%
"""
punto 2 D actualizado usando las maximas distancias entre A y L
"""
#las maximas distancias se encuentran en las columnas [[382],[383],[384],[402],[409],[410],[411]
#,[412],[413],[429],[437],[438],[439](maximo),[440],[465],[466],[467],[468],[494],[495]]

X_train_caso1 = X_dev.iloc[:, 438:440]
X_test_caso1 = X_eval.iloc[:, 438:440]
# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso1, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso1 = model.predict(X_test_caso1) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso1)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.9583333333333334

X_train_caso2 = X_dev.iloc[:, 382:384]
X_test_caso2 = X_eval.iloc[:, 382:384]
# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso2, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso2 = model.predict(X_test_caso2) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso2)
print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.9302083333333333

X_train_caso3 = X_dev.iloc[:, [402,409,410]]
X_test_caso3 = X_eval.iloc[:, [402,409,410]]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso3, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso3 = model.predict(X_test_caso3) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso3)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.9427083333333334

X_train_caso4 = X_dev.iloc[:, [468,494,495]]
X_test_caso4 = X_eval.iloc[:, [468,494,495]]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso4, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso4 = model.predict(X_test_caso4) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso4)

print(f"Accuracy con 3 atributos: {score}")
#Accuracy con 3 atributos: 0.9635416666666666

"""
Ahora voy a tomar probar tomando mas cantidad de atributos, primero voy a tomar 5, luego 10 y finalmente
voy a tomar 15
"""
X_train_caso5 = X_dev.iloc[:, [382,383,384,402,409]]
X_test_caso5 = X_eval.iloc[:, [382,383,384,402,409]]

# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso5, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso5 = model.predict(X_test_caso5) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso5)
print(f"Accuracy con 5 atributos: {score}")
#Accuracy con 5 atributos: 0.965625

X_train_caso6 = X_dev.iloc[:, [409,410,411,412,413,429,437,438,439,440]]
X_test_caso6 = X_eval.iloc[:, [409,410,411,412,413,429,437,438,439,440]]
#409],[410],[411]
#,[412],[413],[429],[437],[438],[439](maximo),[440]
# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso6, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso6 = model.predict(X_test_caso6) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso6)
print(f"Accuracy con 10 atributos: {score}")
#Accuracy con 10 atributos: 0.9822916666666667

X_train_caso7 = X_dev.iloc[:, [410,411,412,413,429,437,438,439,440,465, 466, 467, 468,494,495]]
X_test_caso7 = X_eval.iloc[:, [410,411,412,413,429,437,438,439,440,465, 466, 467, 468,494,495]]
# Ajustamos el modelo KNN

model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train_caso7, y_dev) # entreno el modelo con los datos X e Y
Y_pred_caso7 = model.predict(X_test_caso7) # me fijo qué clases les asigna el modelo a mis datos
score = metrics.accuracy_score(y_eval, Y_pred_caso7)
print(f"Accuracy con 15 atributos: {score}")
#Accuracy con 15 atributos: 0.984375

#%% Punto D-E 
"""
Voy a comparar el ultimo modelo que tiene 20 atributos de distancia maxima en promedio entre L y A
tomando vecinos {k= 2, 5, 10, 15, 20, 50} a ver que sucede
"""

k_values = [2, 5, 10, 15, 20, 50]
accuracy_train7 = []
accuracy_test7 = []

X_train_caso7 = X_dev.iloc[:, [410,411,412,413,429,437,438,439,440,465, 466, 467, 468,494,495]]
X_test_caso7 = X_eval.iloc[:, [410,411,412,413,429,437,438,439,440,465, 466, 467, 468,494,495]]

# Iterar sobre los valores de K
for k in k_values:
    # Ajustar el modelo KNN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_caso7, y_dev)
    
    # Predecir y calcular el accuracy para los datos de entrenamiento
    Y_pred_caso7 = model.predict(X_train_caso7)
    train_accuracy = accuracy_score(y_dev, Y_pred_caso7)
    accuracy_train7.append(train_accuracy)
    
    # Predecir y calcular el accuracy para los datos de prueba
    Y_pred_caso7 = model.predict(X_test_caso7)
    test_accuracy7 = accuracy_score(y_eval, Y_pred_caso7)
    accuracy_test7.append(test_accuracy7)

# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar los datos
plt.plot(k_values, accuracy_train7, label='Train Accuracy', marker='o')
plt.plot(k_values, accuracy_test7, label='Test Accuracy', marker='o')

# Añadir etiquetas y título
plt.xlabel('Valores de K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Valores de K')
plt.legend()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xticks(range(0, max(k_values)+1, 5))  # Ajustar la grilla del eje X a intervalos de 5

# Mostrar el gráfico
plt.show()

# Mostrar los valores de accuracy para train y test
print(f"Accuracy de entrenamiento con 15 atributos con k=2: {accuracy_train7[0]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=5: {accuracy_train7[1]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=10: {accuracy_train7[2]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=20: {accuracy_train7[3]}")
print(f"Accuracy de entrenamiento con 15 atributos con k=50: {accuracy_train7[4]}")
print(f"Accuracy de prueba: {accuracy_test7}")
#%% 2_e
"""
Voy a comparar el ultimo modelo tomando vecinos {k= 2, 5, 10, 15, 20, 50} a ver que sucede.
NO BORRE LA PARTE ESTA VIEJA PORQUE HABRIA QUE ANALIZAR EL GRAFICO, SIENTO QUE ESTE ES MAS LINDO Y NO SE
LA RAZON
"""

k_values = [2, 5, 10, 15, 20, 50]
accuracy_train6 = []
accuracy_test6 = []

# Seleccionar los 20 atributos más cercanos al máximo pixel para el promedio
X_train_caso6 = X_dev.iloc[:, 396:416]
X_test_caso6 = X_eval.iloc[:, 396:416]

# Iterar sobre los valores de K
for k in k_values:
    # Ajustar el modelo KNN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_caso6, y_dev)
    
    # Predecir y calcular el accuracy para los datos de entrenamiento
    Y_pred_caso6 = model.predict(X_train_caso6)
    train_accuracy = accuracy_score(y_dev, Y_pred_caso6)
    accuracy_train6.append(train_accuracy)
    
    # Predecir y calcular el accuracy para los datos de prueba
    Y_pred_caso6 = model.predict(X_test_caso6)
    test_accuracy6 = accuracy_score(y_eval, Y_pred_caso6)
    accuracy_test6.append(test_accuracy6)

# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar los datos
plt.plot(k_values, accuracy_train6, label='Train Accuracy', marker='o')
plt.plot(k_values, accuracy_test6, label='Test Accuracy', marker='o')

# Añadir etiquetas y título
plt.xlabel('Valores de K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Valores de K')
plt.legend()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xticks(range(0, max(k_values)+1, 5))  # Ajustar la grilla del eje X a intervalos de 5

# Mostrar el gráfico
plt.show()

# Mostrar los valores de accuracy para train y test
print(f"Accuracy de entrenamiento con 20 atributos con k=2: {accuracy_train6[0]}")
print(f"Accuracy de entrenamiento con 20 atributos con k=5: {accuracy_train6[1]}")
print(f"Accuracy de entrenamiento con 20 atributos con k=10: {accuracy_train6[2]}")
print(f"Accuracy de entrenamiento con 20 atributos con k=20: {accuracy_train6[3]}")
print(f"Accuracy de entrenamiento con 20 atributos con k=50: {accuracy_train6[4]}")
print(f"Accuracy de prueba: {accuracy_test6}")
#%%
"""
Ahora vamos a tomar 30 atributos  del centro de mi dataframe test (indice 392) e ir 
tomando vecinos iguales a {k= 2, 5, 10, 15, 20, 50} a ver que sucede
"""

X_train_caso7 = X_dev.iloc[:, 377:407]
X_test_caso7 = X_eval.iloc[:, 377:407]

k_values = [2, 5, 10, 15, 20, 50]
accuracy_train7 = []
accuracy_test7 = []


for k in k_values:
    # Ajustar el modelo KNN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_caso7, y_dev)
    
    # Predecir y calcular el accuracy para los datos de entrenamiento
    Y_pred_caso7 = model.predict(X_train_caso7)
    train_accuracy7 = accuracy_score(y_dev, Y_pred_caso7)
    accuracy_train7.append(train_accuracy7)
    
    # Predecir y calcular el accuracy para los datos de prueba
    Y_pred_caso7 = model.predict(X_test_caso7)
    test_accuracy7 = accuracy_score(y_eval, Y_pred_caso7)
    accuracy_test7.append(test_accuracy7)
# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar los datos
plt.plot(k_values, accuracy_train7, label='Train Accuracy', marker='o')
plt.plot(k_values, accuracy_test7, label='Test Accuracy', marker='o')

# Añadir etiquetas y título
plt.xlabel('Valores de K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Valores de K')
plt.legend()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xticks(range(0, max(k_values)+1, 5))  # Ajustar la grilla del eje X a intervalos de 5

# Mostrar el gráfico
plt.show()

# Mostrar los valores de accuracy para train y test
print(f"Accuracy de entrenamiento con 30 atributos con k=2: {accuracy_train7[0]}")
print(f"Accuracy de entrenamiento con 30 atributos con k=5: {accuracy_train7[1]}")
print(f"Accuracy de entrenamiento con 30 atributos con k=10: {accuracy_train7[2]}")
print(f"Accuracy de entrenamiento con 30 atributos con k=20: {accuracy_train7[3]}")
print(f"Accuracy de entrenamiento con 30 atributos con k=50: {accuracy_train7[4]}")
print(f"Accuracy de prueba: {accuracy_test7}")

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

















