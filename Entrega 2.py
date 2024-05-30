import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
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

# Apply flip_rotate function to each image
transformed_images = np.array([flip_rotate(image) for image in images])

# Flatten the transformed images back to 1D
flattened_images = transformed_images.reshape(transformed_images.shape[0], -1)

# Create a new DataFrame with the labels and transformed images
transformed_data = pd.DataFrame(np.column_stack((labels, flattened_images)))
data1 = transformed_data.copy()
#%%
#cuento cantidad de letras
cantidad_letras = data1[0].value_counts()
# indices = data1[0][data1[0] == 'M']
#%%
matriz_valores = np.array(transformed_data.drop(0, axis=1).astype(float))


vec_promedio = np.mean(matriz_valores, axis=0)

image_vec_promedio = vec_promedio.reshape(28, 28)

plt.imshow(image_vec_promedio, cmap='gray')
plt.title("Average Image")
plt.show()
#%%
#PUNTO 2
data1_frame_la = data1[(data1[0] == 'L') | (data1[0] == 'A')]
cantidad_por_letra = data1_frame_la[0].value_counts()
X_dev, X_eval, y_dev, y_eval = train_test_split(data1_frame_la.drop(0, axis =1),data1_frame_la[0],random_state=1,test_size=0.2)


#%%
#PUNTO 3 
data1_frame_vocales = data1[ (data1[0] == 'A') | (data1[0] == 'E') | (data1[0] == 'I') | (data1[0] == 'O') | (data1[0] == 'U')]
X_dev1, X_eval1, y_dev1, y_eval1 = train_test_split(data1_frame_vocales.drop(0, axis =1),data1_frame_vocales[0],random_state=1,test_size=0.2)
