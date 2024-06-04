"""
Trabajo Práctico 01
Materia: Laboratorio de datos - FCEyN - UBA
Integrantes: Otermín Juana, Quispe Rojas Luis Enrique , Vilcovsky Maia

Fecha  : 2024-06-04
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import euclidean
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

def crear_arbol(profundidad, X, Y):
    clf_info = DecisionTreeClassifier(max_depth=profundidad)
    clf_info = clf_info.fit(X, Y)
    return clf_info

#%%
#esta parte rota y flipea todas las letras en el data frame
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values

transformed_images = np.array([flip_rotate(image) for image in images])
flattened_images = transformed_images.reshape(transformed_images.shape[0], -1)
data1 = pd.DataFrame(np.column_stack((labels, flattened_images)))

#cuento cantidad de letras
cantidad_letras = data1[0].value_counts()
#%%
matriz_valores = np.array(data1.drop(0, axis=1).astype(float))
vec_promedio = np.mean(matriz_valores, axis=0)
image_vec_promedio = vec_promedio.reshape(28, 28)

plt.figure(figsize=(17.28, 8.1))
plt.imshow(image_vec_promedio)
plt.title("Promedio de todas las letras")
plt.show()
#%%
# Encontrar las columnas con valores promedio entre 0 y 1
columnas_menores_1 = np.where((vec_promedio >= 0) & (vec_promedio <= 1))[0]

# Transformar el vector promedio en una imagen
imagen_vec_promedio = vec_promedio.reshape(28, 28)

# Visualizar la imagen promedio con recuadros en las columnas de bajo valor
plt.figure(figsize=(17.28, 8.1))
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

plt.figure(figsize=(17.28, 8.1))
sns.heatmap(distances_normalized, xticklabels=selected_letters, yticklabels=selected_letters, cmap='viridis', annot=True, fmt=".2f")
plt.title('Mapa de calor de las distancias normalizadas entre imágenes promedio de letras seleccionadas')
plt.xlabel('Letra')
plt.ylabel('Letra')
plt.show()

#%%
#NUEVO PTO 1C
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

# Encontrar la distancia mínima y máxima
min_distance = min(avg_distances.values())
max_distance = max(avg_distances.values())

# Normalizar las distancias promedio
normalized_distances = {letter: (distance - min_distance) / (max_distance - min_distance) for letter, distance in avg_distances.items()}

# Configurar el gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(normalized_distances.keys(), normalized_distances.values(), color='skyblue')

# Añadir los valores normalizados encima de cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')
    
# Añadir una línea vertical en la distancia promedio especificada (normalizada)
normalized_threshold = (1799 - min_distance) / (max_distance - min_distance)
plt.axhline(y=normalized_threshold, color='r', linestyle='--', label=f'Promedio normalizado = {normalized_threshold:.4f}')
plt.legend()
plt.xlabel('Letras')
plt.ylabel('Distancia Promedio Normalizada')
plt.title('Distancia Promedio Normalizada de cada tipo de letra a su promedio')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

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
promedios_L = np.mean(data1_frame_la[data1_frame_la.loc[:,0] == "A"].loc[:,1:], axis=0)
promedios_A = np.mean(data1_frame_la[data1_frame_la.loc[:,0] == "L"].loc[:,1:], axis =0)

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
#PUNTO 3 
data1_frame_vocales = data1[ (data1[0] == 'A') | (data1[0] == 'E') | (data1[0] == 'I') | (data1[0] == 'O') | (data1[0] == 'U')]
X_dev1, X_eval1, y_dev1, y_eval1 = train_test_split(data1_frame_vocales.drop(0, axis =1),data1_frame_vocales[0],random_state=1,test_size=0.2)
max_depths = range(1, 11)
accuracies = []



for depth in max_depths:
    clf = crear_arbol(depth, X_dev1, y_dev1)
    y_pred = clf.predict(X_eval1)
    accuracy = accuracy_score(y_eval1, y_pred)
    accuracies.append(accuracy)


accuracy_deltas = np.diff(accuracies)
elbow_depth = 5
best_depth = elbow_depth
best_tree = crear_arbol(best_depth, X_dev1, y_dev1)
best_accuracy = accuracies[best_depth - 1]

print(f'Elbow Depth: {best_depth}, Accuracy at Elbow: {best_accuracy}')

# Optionally, plot the accuracies to visualize the elbow point
plt.plot(max_depths, accuracies, marker='o')
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Depth of Decision Tree')
plt.axvline(x=elbow_depth, color='r', linestyle='--')
plt.show()
#%%
# Visualizing the best tree
feature_names = X_dev1.columns.tolist()
class_names = y_dev1.unique().tolist()
param_grid = {
    'max_depth': [5],
    'criterion': ['gini', 'entropy']
}

kf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, scoring='accuracy')

# Fit the model
kf.fit(X_dev1, y_dev1)

# Get the best parameters and best score
best_params = kf.best_params_
best_score = kf.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score}')
#%%
# Train the best model on the full training set
best_tree = DecisionTreeClassifier(**best_params)
best_tree.fit(X_dev1, y_dev1)

# Visualize the tree using matplotlib
plt.figure(figsize=(20, 10))
plot_tree(best_tree, feature_names=X_dev1.columns, class_names=y_dev1.unique().astype(str), filled=True, rounded=True, fontsize=5)
plt.show()

#%%
#test en los holdout
heldout_predictions = best_tree.predict(X_eval1)
sum(heldout_predictions == y_eval1)

cm = confusion_matrix(y_eval1, heldout_predictions, labels=y_eval1.unique())

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_eval1.unique())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Vowels Classification')
plt.show()

accuracy = accuracy_score(y_eval1, heldout_predictions)
precision = precision_score(y_eval1, heldout_predictions, labels=y_eval1.unique(), average='macro')
recall = recall_score(y_eval1, heldout_predictions, labels=y_eval1.unique(), average='macro')
f1 = f1_score(y_eval1, heldout_predictions, labels=y_eval1.unique(), average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
