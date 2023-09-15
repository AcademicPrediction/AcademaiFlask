from flask import Flask, request, send_file
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def create_app():
    # Realizar la configuración aquí si es necesario
    return app

def ajustar_valores(valor):
    try:
        valor = float(valor)
        return max(0, min(20, valor))
    except:
        return 0

def procesar_hoja(df, nombre_hoja):
    if df.empty:
        return f"Error: La hoja '{nombre_hoja}' está completamente vacía."
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(ajustar_valores)
    return df

def guardar_en_excel(df_predicciones):
    excel_path = 'temporary_file.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_predicciones.to_excel(writer, sheet_name="ACADEMAI", index=False)
        
        for column in writer.sheets["ACADEMAI"].columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            writer.sheets["ACADEMAI"].column_dimensions[column[0].column_letter].width = adjusted_width
    return excel_path

def build_optimized_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

@app.route('/generate_excel', methods=['POST'])
def predict():
    modelo_guardado_path = 'my_model.h5'
    model_existente = os.path.exists(modelo_guardado_path)

    if model_existente:
        print("Cargando modelo existente...")
        model = tf.keras.models.load_model(modelo_guardado_path)

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    hojas = pd.read_excel(file, sheet_name=None, engine='openpyxl')
    dfs = [procesar_hoja(df, nombre_hoja) for nombre_hoja, df in hojas.items()]

    df_predicciones = dfs[-1][['Alumno']].copy()

    for curso in dfs[0].columns[1:]:
        print(f"Entrenando modelo para: {curso}")

        X = []
        y = []

        for idx, df in enumerate(dfs[:-1]):
            features = df.drop(columns=["Alumno"]).values.tolist()
            targets = dfs[idx+1][curso].values
            X.extend(features)
            y.extend(targets)

        X = np.array(X)
        y = np.array(y)

        if model_existente and model.input_shape[1] == X.shape[1]:
            pass
        else:
            print("Creando un nuevo modelo...")
            model = build_optimized_model((X.shape[1],))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        model_checkpoint = ModelCheckpoint(modelo_guardado_path, save_best_only=True, monitor='val_loss', mode='min')

        callbacks_list = [early_stopping, reduce_lr, model_checkpoint]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks_list)

        predicciones = model.predict(dfs[-1].drop(columns=["Alumno"]).values)
        predicciones = np.clip(predicciones, 0, 20)
        df_predicciones[curso + ' Predicted'] = predicciones

    for curso in df_predicciones.columns[1:]:
        df_predicciones[curso] = df_predicciones[curso].apply(lambda x: round(x, 2))

    excel_path = guardar_en_excel(df_predicciones)
    return send_file(excel_path, as_attachment=True)