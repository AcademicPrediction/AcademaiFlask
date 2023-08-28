from flask import Flask, request, send_file
import pandas as pd
import joblib

application = app = Flask(__name__)

model = joblib.load('model.pkl')


@app.route('/generate_excel', methods=['POST'])
def generate_excel():
    
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']

    # Si el usuario no selecciona el archivo, el navegador envía un archivo vacío sin nombre.
    if file.filename == '':
        return 'No selected file', 400

    # Leer el archivo Excel en un DataFrame
    df1 = pd.read_excel(file, sheet_name='Grado x')
    nombres_cursos = df1.columns.tolist()
    caracteristicas_prediccion = df1.values

    predicciones = model.predict(caracteristicas_prediccion)
    df_predicciones = pd.DataFrame(predicciones, columns=nombres_cursos)
    
    # Guardamos el DataFrame en un archivo Excel
    excel_path = 'temporary_file.xlsx'
    df_predicciones.to_excel(excel_path, index=False)
    
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000)
    app.run(debug=True)
