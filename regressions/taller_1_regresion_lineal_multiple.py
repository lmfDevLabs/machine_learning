# Nombre: Taller 1 - Regresión Lineal Múltiple
# Autor: Ruben Romero y Carlos A. Talero J
# Fecha: 2021-09-01
# Descripción: Implementación de un modelo de regresión lineal múltiple para predecir el costo de seguro médico de pacientes.
# Programa: Maestría de Inteligencia Artificial
# Universidad: Pontificia Universidad Javeriana
# Ciudad: Bogotá, D.C., Colombia
# Profesor: Ing. Andres Dario Moreno Barbosa
# Materia: Aprendizaje de Máquina

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Creación de objeto pandas dataframe
patients_df=pd.read_csv('https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv?raw=true')

# Creación variables binarias para sex y smoker
patients_df.replace({'sex':{'male':0,'female':1}}, inplace=True)
patients_df.replace({'smoker':{'yes':1,'no':0}}, inplace=True)

# logaritmo natural para bmi
# patients_df['bmi'] = np.log(patients_df['bmi'])

# Histograma de BMI transformado
# plt.figure(figsize=[8, 6])  
# plt.hist(patients_df['bmi'], bins=30, color='blue', alpha=0.7)  
# plt.title('Distribución de log(BMI)')  
# plt.xlabel('log(BMI)')  
# plt.ylabel('Frecuencia')  
# plt.grid(True)  
# plt.show()  

# Creación de variables dummies
region_dummies_df=pd.get_dummies(patients_df[['region']])

# Unión de variables dummies al dataframe
patients_df = patients_df.join(region_dummies_df)

# crear variable binaria para malos hábitos
patients_df['bad_habits'] = (
        (patients_df['age'] > 50) | 
        (patients_df['bmi'] < 18) | 
        (patients_df['bmi'] > 30) | 
        (patients_df['smoker'] == 1)
    ).astype(int)

# dummy variables a binarias para region
def convert_bool_to_binary(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(int)
    return df

columns_to_convert = ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
patients_df = convert_bool_to_binary(patients_df, columns_to_convert) 
print(patients_df.head())   

# Conteo de observaciones por región
region_counts = patients_df[['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']].sum()

# Gráfico de barras para las regiones
plt.figure(figsize=[10, 6])  
plt.bar(region_counts.index, region_counts.values, color=['blue', 'green', 'red', 'purple'])  
plt.title('Número de Observaciones por Región')  
plt.xlabel('Región')  
plt.ylabel('Número de Observaciones')  
plt.xticks(rotation=45) 
plt.show()  

# MSE
def calculate_mse(X, y, theta):
    m = len(y)
    y_pred = X.dot(theta)
    error = y_pred - y
    mse = (1/(2*m)) * np.sum(error**2)
    return mse

# Gradiente descendente
def gradient_descent(X, y, theta, learning_rate, iterations, threshold):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.dot(X.transpose(), (predictions - y))
        theta -= learning_rate * (1/m) * errors
        cost_history[i] = calculate_mse(X, y, theta)
        if i > 0 and abs(cost_history[i] - cost_history[i-1]) < threshold:
            print(f"Convergencia alcanzada en la iteración {i}.")
            break
    return theta, cost_history[:i+1]  # Devuelve theta y el historial de costo hasta la convergencia.

# Preparación de datos
def prepare_data(df, feature_names, label_name):
    # Selecciona múltiples columnas de características basadas en feature_names
    X = df[feature_names].values
    # Infla X con una columna de unos al inicio para el término de intercepción
    X_c = np.hstack((np.ones((X.shape[0], 1)), X))
    # Aplica logaritmo natural a la variable dependiente
    Y = df[[label_name]].values
    # logaritmo natural para charges
    Y = np.log(Y)
    # Retorna el diccionario con las matrices X y Y preparadas
    return {'X': X_c, 'Y': Y}

#feature_names = ['age', 'bmi', 'children']
#feature_names = ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
#feature_names = ['sex','smoker']
#feature_names = ['age','bad_habits',]
feature_names = ['age','smoker','bmi',"bad_habits"]
label_name = 'charges'

# Uso 70% para entrenamiento (random split)
train_df= patients_df.sample(frac=0.7,random_state=200)
rest_df= patients_df.drop(train_df.index)
# Uso 15% para validacion y 15% para test
val_df=rest_df.sample(frac=0.5,random_state=200)
test_df=rest_df.drop(val_df.index)

# Preparar los datos de entrenamiento, validación y prueba
train_data = prepare_data(train_df, feature_names, label_name)
val_data = prepare_data(val_df, feature_names, label_name)
test_data = prepare_data(test_df, feature_names, label_name)

# Parámetros del algoritmo (posibles experimentos)
learning_rate = 0.0001
iterations = 10000
threshold = 0.001
theta = np.random.rand(len(feature_names) + 1, 1)  # +1 por el término de intercepción

# Entrenar usando la edad
theta_optimal, _ = gradient_descent(train_data['X'], train_data['Y'], theta, learning_rate, iterations, threshold)

# Visualización del historial de costo
plt.figure(figsize=[8,6])
plt.plot(_,'r')  # '_' contiene el historial de costo
plt.grid(True)
plt.title('Convergencia de la Función de Costo')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.show()

# Cálculo del MSE para los conjuntos de datos
mse_train = calculate_mse(train_data['X'],train_data['Y'],theta_optimal)
mse_val = calculate_mse(val_data['X'],val_data['Y'],theta_optimal)
mse_test = calculate_mse(test_data['X'],test_data['Y'],theta_optimal)

# Impresión de resultados
print("MSE en entrenamiento:", mse_train)
print("MSE en validación:", mse_val)
print("MSE en prueba:", mse_test)

# Función para visualizar valores reales vs. predicciones
def plot_real_vs_predicted(X, y, theta, title='Valores Reales vs. Predicciones'):
    y_pred = X.dot(theta)
    plt.figure(figsize=[8,6])
    plt.scatter(y, y_pred, alpha=0.5)
    plt.title(title)
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.show()

# Visualizar valores reales vs. predicciones para entrenamiento, validación y prueba
plot_real_vs_predicted(train_data['X'], train_data['Y'], theta_optimal, title='Entrenamiento: Valores Reales vs. Predicciones')
plot_real_vs_predicted(val_data['X'], val_data['Y'], theta_optimal, title='Validación: Valores Reales vs. Predicciones')
plot_real_vs_predicted(test_data['X'], test_data['Y'], theta_optimal, title='Prueba: Valores Reales vs. Predicciones')


