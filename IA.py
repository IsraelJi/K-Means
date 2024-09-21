import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# Verificar si el archivo existe antes de intentar leerlo
ruta_archivo = r'C:/Users/crazy/OneDrive/Documentos/ESCUELA/SEM_5/IA/DataAlum/DataAlum.csv'
alumnos = pd.read_csv(ruta_archivo, engine='python', encoding='latin1')


# Mostrar información del DataFrame
alumnos.info()
print(alumnos.head())

# Eliminar la columna 'Nombre'
alumnos_variable = alumnos.drop(['Nombre'], axis=1)

# Normalizar los datos
alumnos_normalizado = (alumnos_variable - alumnos_variable.min()) / (alumnos_variable.max() - alumnos_variable.min())

# Aplicar K-Means
clustering = KMeans(n_clusters=2, max_iter=300, random_state=42)
clustering.fit(alumnos_normalizado)

# Asignar los clusters al DataFrame original
alumnos['KMeans_Clusters'] = clustering.labels_

# Imprimir las etiquetas de los clusters
print("Etiquetas de los clusters:")
print(alumnos['KMeans_Clusters'].value_counts())

# Asegurar que los clusters 0 y 1 estén siempre representados por los mismos colores
color_map = {0: 'blue', 1: 'green'}
alumnos['Color'] = alumnos['KMeans_Clusters'].map(color_map)

# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=2)
alumnos_pca = pca.fit_transform(alumnos_normalizado)
alumnos_pca_df = pd.DataFrame(data=alumnos_pca, columns=['Componente_1', 'Componente_2'])
alumnos_pca_nombre = pd.concat([alumnos_pca_df, alumnos[['KMeans_Clusters', 'Color']]], axis=1)



# Graficar los resultados
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Componentes Principales', fontsize=20)

# Usar los colores definidos en el mapa de colores
colors = alumnos_pca_nombre['Color'].values
ax.scatter(x=alumnos_pca_nombre.Componente_1, y=alumnos_pca_nombre.Componente_2, c=colors, s=50)

# Mostrar la gráfica
plt.show()
