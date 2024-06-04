import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Cargar el dataset
file_path = '/Users/carlosalbertotalerojacome/Documents/dev/ML_Final/assets/InstaCart 1.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
df.drop('eval_set', axis=1, inplace=True)

# Convertir los nombres de producto en enteros consecutivos
product_encoder = LabelEncoder()
df['product_name'] = df['product_name'].astype(str)  # Asegúrate de que 'product_name' sea una columna en tu DataFrame
df['product_name'] = product_encoder.fit_transform(df['product_name'])

# Agrupar por usuario y orden para crear secuencias
user_orders = df.sort_values(['user_id', 'order_id', 'add_to_cart_order']).groupby(['user_id', 'order_id'])['product_name'].apply(list)

# Crear secuencias de entrada y etiquetas
sequence_length = 3
X = []
y = []

for order_sequence in user_orders:
    if len(order_sequence) > sequence_length:
        for i in range(len(order_sequence) - sequence_length):
            X.append(order_sequence[i:i+sequence_length])
            y.append(order_sequence[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una matriz de interacciones usuario-ítem
interactions = df.pivot_table(index='user_id', columns='product_name', values='reordered', fill_value=0)

# Escalar la matriz de interacciones
scaler = StandardScaler()
interactions_scaled = scaler.fit_transform(interactions)

# Realizar la descomposición SVD
U, S, VT = np.linalg.svd(interactions_scaled, full_matrices=False)

# Seleccionar el número de componentes latentes (k)
k = 50
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

# Reconstruir la matriz reducida
interactions_reduced = np.dot(np.dot(U_k, S_k), VT_k)

# Crear secuencias de entrenamiento
X = []
y = []

for i in range(interactions_reduced.shape[0]):
    for j in range(interactions_reduced.shape[1] - sequence_length):
        X.append(interactions_reduced[i, j:j + sequence_length])
        y.append(interactions_reduced[i, j + sequence_length])

X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Redimensionar los datos para LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Añadir una dimensión adicional
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))  # Añadir una dimensión adicional

# Definir los parámetros de entrada
num_unique_products = len(product_encoder.classes_)

# Iniciar el modelo
model = Sequential()

# Añadir una capa de Embedding
model.add(Input(shape=(sequence_length,)))
model.add(Embedding(input_dim=num_unique_products + 1, output_dim=50))

# Añadir capas LSTM
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Añadir capas densas
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))

# Añadir la capa de salida
model.add(Dense(num_unique_products, activation='softmax'))

# Definir el optimizador
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Compilar el modelo
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Guardar el modelo y otros objetos
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

model.save(os.path.join(output_dir, 'svm_lstm_model.h5'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
joblib.dump(product_encoder, os.path.join(output_dir, 'product_encoder.joblib'))

print("Model and other objects saved successfully.")