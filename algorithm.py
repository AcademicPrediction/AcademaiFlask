import os
import pandas as pd
import numpy as np
from tkinter import filedialog, Tk
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import webbrowser
from openpyxl.worksheet.dimensions import DimensionHolder, ColumnDimension

def seleccionar_archivo_excel():
    """Permite al usuario seleccionar un archivo Excel y devuelve la ruta del archivo."""
    root = Tk()
    root.withdraw()
    ruta_archivo = filedialog.askopenfilename(title="Seleccione un archivo Excel", filetypes=[("Archivos Excel", "*.xls;*.xlsx")])
    return ruta_archivo

def ajustar_valores(valor):
    """Convierte el valor a float y lo ajusta en el rango [0, 20]. Si no es convertible, devuelve 0."""
    try:
        valor = float(valor)
        return max(0, min(20, valor))
    except:
        return 0

def procesar_hoja(df, nombre_hoja):
    """Procesa la hoja del Excel ajustando los valores de las notas."""
    if df.empty:
        return f"Error: La hoja '{nombre_hoja}' está completamente vacía."

    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(ajustar_valores)
    return df

def guardar_en_excel(df_predicciones):
    """Guarda el DataFrame de predicciones en un archivo Excel y lo abre automáticamente."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Archivos Excel", "*.xlsx")])
    
    if not file_path:
        return

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_predicciones.to_excel(writer, sheet_name="ACADEMAI", index=False)
        
        # Ajustar automáticamente el tamaño de las columnas
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

    webbrowser.open(file_path)

# Selecciona el archivo Excel con las notas
nombre_archivo = seleccionar_archivo_excel()
if nombre_archivo:
    _, extension = os.path.splitext(nombre_archivo)
    if extension not in ['.xls', '.xlsx']:
        print("Error: El archivo no es de tipo Excel.")
    else:
        try:
            # Leer todas las hojas del archivo Excel
            hojas = pd.read_excel(nombre_archivo, sheet_name=None, engine='openpyxl')
            dfs = []

            # Procesar cada hoja ajustando los valores
            for nombre_hoja, df in hojas.items():
                print(f"Procesando hoja: {nombre_hoja}")
                df = procesar_hoja(df, nombre_hoja)
                dfs.append(df)

            # Crear un DataFrame para las predicciones con la columna 'Alumno'
            df_predicciones = dfs[-1][['Alumno']].copy()

            # Entrenar el modelo y predecir para cada curso
            for curso in dfs[0].columns[1:]:
                print(f"Entrenando modelo para: {curso}")

                X = []
                y = []

                # Recopilar datos de entrenamiento: Características y objetivos
                for idx, df in enumerate(dfs[:-1]):
                    features = df.drop(columns=["Alumno"]).values.tolist()
                    targets = dfs[idx+1][curso].values
                    X.extend(features)
                    y.extend(targets)

                X = np.array(X)
                y = np.array(y)
                
                # Dividir datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Configuración de la red neuronal profunda
                model = keras.Sequential([
                    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Aplicar detención temprana para evitar el sobreajuste
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
                
                model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
                
                # Predecir para el último grado ingresado
                predicciones = model.predict(dfs[-1].drop(columns=["Alumno"]).values)
                predicciones = np.clip(predicciones, 0, 20)  # Asegurar valores en el rango [0,20]
                predicciones = predicciones.round(2)  # Asegurarse de que esté redondeado a 2 decimales
                df_predicciones[curso + ' Predicted'] = predicciones

            # Guardar predicciones en un nuevo archivo Excel
            guardar_en_excel(df_predicciones)

        except Exception as e:
            print(f"Error al leer el archivo: {e}")
