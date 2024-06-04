# flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# basics
import numpy as np
import pandas as pd
# sk
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
# gcp
from google.cloud import storage
# others
import joblib

# flask app
app = Flask(__name__)
CORS(app)

# GCP
# Configuración del bucket y los archivos del modelo
BUCKET_NAME = 'your-bucket-name'
MODEL_FILES = ['knn_model.joblib', 'scaler.joblib', 'label_encoder.joblib']
# Descargar los archivos del modelo desde GCS
def download_model_files(bucket_name, model_files):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for model_file in model_files:
        blob = bucket.blob(f'models/{model_file}')
        blob.download_to_filename(model_file)
        print(f'Downloaded {model_file} from bucket.')

# Cargar los modelos desde los archivos descargados o locales 
# KNN
knn_recommendation_model = joblib.load('./models/knn/knn_model.joblib')
scaler = joblib.load('./models/knn/scaler.joblib')
le_product = joblib.load('./models/knn/label_encoder.joblib')
user_product_matrix_scaled = joblib.load('./models/knn/user_product_matrix_scaled.joblib')

# SVV - LSTM
model = load_model('models/svm_lstm/svm_lstm_model.h5')
scaler = joblib.load('models/svm_lstm/scaler.joblib')
product_encoder = joblib.load('models/svm_lstm/product_encoder.joblib')

# route test
@app.route('/test', methods=['POST'])
def test_route():
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Devolver los datos recibidos
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# products
@app.route('/products', methods=['GET'])
def get_products():
    product_names = product_encoder.classes_
    return jsonify({"products": list(product_names)})

# route knn
@app.route('/knn', methods=['POST'])
def predict_knn():
    data = request.get_json()
    product_names = data['product_names']
    n_recommendations = data.get('n_recommendations', 5)

    try:
        product_ids = le_product.transform(product_names)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    recommendations = []
    for product_id in product_ids:
        distances, indices = knn_recommendation_model.kneighbors(
            user_product_matrix_scaled[:, product_id].reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )
        recommended_product_ids = indices.flatten()[1:]  # Excluimos el primer producto que es el mismo
        recommendations.extend(recommended_product_ids)

    # Filtrar duplicados y limitar al número de recomendaciones deseado
    recommendations = list(set(recommendations))[:n_recommendations]
    recommended_product_names = le_product.inverse_transform(recommendations)
    return jsonify({"recommended_products": recommended_product_names.tolist()})

# Ruta svm_lstm
@app.route('/svm_lstm', methods=['POST'])
def predict_svm_lstm():
    data = request.get_json()
    product_names = data['product_names']
    n_recommendations = data.get('n_recommendations', 5)

    try:
        product_ids = product_encoder.transform(product_names)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Crear la secuencia de entrada
    sequence_length = 3  # Ajusta según la longitud de secuencia que usaste para entrenar
    input_sequence = np.zeros((1, sequence_length, 1))

    for i in range(min(sequence_length, len(product_ids))):
        input_sequence[0, i, 0] = product_ids[i]

    # Predecir con el modelo
    predictions = model.predict(input_sequence)
    recommended_product_ids = predictions[0].argsort()[-n_recommendations:][::-1]

    recommended_product_names = product_encoder.inverse_transform(recommended_product_ids)
    return jsonify({"recommended_products": recommended_product_names.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)