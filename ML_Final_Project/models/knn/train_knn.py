import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Cargar el dataset
file_path = '/Users/carlosalbertotalerojacome/Documents/dev/ML_Final/assets/InstaCart 1.xlsx'
data = pd.read_excel(file_path)
data.drop('eval_set', axis=1, inplace=True)
print(data.head(5))

# Crear la matriz usuario-producto
user_product_matrix = data.pivot_table(index='user_id', columns='product_id', values='order_id', aggfunc='size', fill_value=0)
user_product_matrix = user_product_matrix.applymap(lambda x: 1 if x > 0 else 0)

# Escalar la matriz de usuario-producto
scaler = StandardScaler()
user_product_matrix_scaled = scaler.fit_transform(user_product_matrix)

# Entrenar un modelo KNN para recomendaciones de productos
knn_recommendation_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_recommendation_model.fit(user_product_matrix_scaled.T)  # Transponemos la matriz para tener productos por usuarios

# Guardar el modelo y otros objetos
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

joblib.dump(knn_recommendation_model, os.path.join(output_dir, 'knn_model.joblib'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

# Crear LabelEncoder para los nombres de los productos
product_names = data['product_name'].unique()  # Asegúrate de tener una columna 'product_name' en tu dataset
le_product = LabelEncoder()
le_product.fit(product_names)
joblib.dump(le_product, os.path.join(output_dir, 'label_encoder.joblib'))

# Guardar la matriz de usuario-producto escalada
joblib.dump(user_product_matrix_scaled, os.path.join(output_dir, 'user_product_matrix_scaled.joblib'))

print("Model and other objects saved successfully.")