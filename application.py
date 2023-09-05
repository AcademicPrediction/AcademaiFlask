from flask import Flask, request, send_file
import pandas as pd
import numpy as np
from keras.models import load_model  # Importa la función para cargar modelos

application = app = Flask(__name__)

# Cambio aquí: usar la función load_model para cargar el modelo
model = load_model('my_single_model.h5')  # Asumiendo que el modelo está guardado como 'model.h5'

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
        df = procesar_hoja(df, nombre_hoja)
        dfs.append(df)
    
    # Aquí es donde haces tus predicciones
    df_predicciones = dfs[-1][['Alumno']].copy()

    for curso in dfs[0].columns[1:]:
        predicciones = model.predict(dfs[-1].drop(columns=["Alumno"]).values.astype(np.float32))
        predicciones = np.clip(predicciones, 0, 20)
        df_predicciones[curso + ' Predicted'] = predicciones.flatten()
        
        print(predicciones)
        print(df_predicciones)

    for curso in df_predicciones.columns[1:]:
        df_predicciones[curso] = df_predicciones[curso].apply(lambda x: round(x, 2))

    # Guardar el DataFrame de predicciones en un archivo Excel temporal
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

if __name__ == '__main__':
    app.run(port=8000)
    app.run(debug=True)