from flask import Flask, request, send_file
import pandas as pd
import numpy as np
from keras.models import load_model  # Importa la función para cargar modelos

application = app = Flask(__name__)

# Cambio aquí: usar la función load_model para cargar el modelo
model = load_model('my_model.h5')  # Asumiendo que el modelo está guardado como 'model.h5'

@app.route('/generate_excel', methods=['POST'])
def generate_excel():
    
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    hojas = pd.read_excel(file, sheet_name=None, engine='openpyxl')
    dfs = []
    
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
        
        
    for curso in df_predicciones.columns[1:]:
        df_predicciones[curso] = df_predicciones[curso].apply(lambda x: round(x, 2))
    
    excel_path = 'temporary_file.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
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
    
    return send_file(excel_path, as_attachment=True)

def procesar_hoja(df, nombre_hoja):
    """Procesa la hoja del Excel ajustando los valores de las notas."""
    if df.empty:
        return f"Error: La hoja '{nombre_hoja}' está completamente vacía."

    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(ajustar_valores)
    return df

def ajustar_valores(valor):
    """Convierte el valor a float y lo ajusta en el rango [0, 20]. Si no es convertible, devuelve 0."""
    try:
        valor = float(valor)
        return max(0, min(20, valor))
    except:
        return 0
    
def guardar_en_excel(df_predicciones):
    """Guarda el DataFrame de predicciones en un archivo Excel y lo abre automáticamente."""
    file_path = "resultados_predicciones.xlsx"

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

if __name__ == '__main__':
    app.run(port=8000)
    app.run(debug=True)
