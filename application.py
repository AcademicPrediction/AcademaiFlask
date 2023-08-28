from flask import Flask, request, send_file
import pandas as pd
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

    df1 = pd.read_excel(file, sheet_name='Grado x')
    nombres_cursos = df1.columns.tolist()
    caracteristicas_prediccion = df1.values

    predicciones = model.predict(caracteristicas_prediccion)
    df_predicciones = pd.DataFrame(predicciones, columns=nombres_cursos)
    
    excel_path = 'temporary_file.xlsx'
    df_predicciones.to_excel(excel_path, index=False)
    
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000)
    app.run(debug=True)
