import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os



# Verificar si el archivo existe antes de intentar leerl
alumnos = pd.read_csv(r'C:/Users/crazy/OneDrive/Documentos/ESCUELA/SEM_5/IA/DataAlum/DataAlum.csv', engine='python', encoding='latin1')

alumnos.info()
alumnos.head()

alumnos_variable = alumnos.drop(['Nombre'], axis=1)

alumnos_variable.describe()

alumnos_normalizado = (alumnos_variable - alumnos_variable.min()) / (alumnos_variable.max() - alumnos_variable.min())
alumnos_normalizado

alumnos_normalizado.describe()

clustering = KMeans(n_clusters=2, max_iter=300)
clustering.fit(alumnos_normalizado)

arr = np.array([[20,5,91,88,85,89,98,90.2], [21,7,83,73,92,84,84,83.2]])

KMeans( copy_x=True, init=arr, max_iter=300,
         n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0) 

alumnos['KMeans_Clusters'] = clustering.labels_
alumnos.head()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
alumnos_pca = pca.fit_transform(alumnos_normalizado)
alumnos_pca_df = pd.DataFrame(data=alumnos_pca, columns=['Componente_1', 'Componente_2'])
alumnos_pca_nombre = pd.concat([alumnos_pca_df, alumnos[['KMeans_Clusters']]], axis=1)

alumnos_pca_nombre

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Componentes Principales', fontsize=20)

color_theme = np.array(['blue', 'green','orange'])
ax.scatter(x=alumnos_pca_nombre.Componente_1, y=alumnos_pca_nombre.Componente_2, c=color_theme[alumnos_pca_nombre.KMeans_Clusters], s=50)

plt.show()
