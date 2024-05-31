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

#%%
#PUNTO 3 
data1_frame_vocales = data1[ (data1[0] == 'A') | (data1[0] == 'E') | (data1[0] == 'I') | (data1[0] == 'O') | (data1[0] == 'U')]
X_dev1, X_eval1, y_dev1, y_eval1 = train_test_split(data1_frame_vocales.drop(0, axis =1),data1_frame_vocales[0],random_state=1,test_size=0.2)
