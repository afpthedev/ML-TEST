from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)  # CORS'u tüm rotalar için etkinleştir

# Tanımlı algoritmalar listesi
ALGORITHMS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Neural Network (MLP)': MLPRegressor(max_iter=1000),
    'AdaBoost Regressor': GradientBoostingRegressor()
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Çalışıyor'})

# Dosya yükleme endpointi
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'}), 400

    try:
        df = pd.read_excel(file)
        columns = df.columns.tolist()
        df.to_excel('uploaded_file.xlsx', index=False)
        return jsonify({'columns': columns})
    except Exception as e:
        return jsonify({'error': f'Dosyayı işlerken bir hata oluştu: {str(e)}'}), 500

# Çoklu makine öğrenmesi modeli analiz endpointi
@app.route('/analyze', methods=['POST'])
def analyze_data():
    data = request.get_json()
    dependent_variable = data.get('dependent_variable')

    if not dependent_variable:
        return jsonify({'error': 'Bağımlı değişken belirtilmedi.'}), 400

    try:
        df = pd.read_excel('uploaded_file.xlsx')
        if dependent_variable not in df.columns:
            return jsonify({'error': f'Bağımlı değişken {dependent_variable} veri setinde bulunamadı.'}), 400

        df = pd.get_dummies(df, drop_first=True)
        X = df.drop(columns=[dependent_variable])
        y = df[dependent_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        results = {}
        for model_name, model in ALGORITHMS.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                results[model_name] = {
                    'mse': round(mse, 4),
                    'r2_score': round(r2, 4),
                    'predictions': predictions.tolist(),
                    'actual_values': y_test.tolist()
                }
            except Exception as e:
                results[model_name] = {'error': str(e)}

        return jsonify(results)
    except Exception as e:
        error_message = f'Analiz sırasında bir hata oluştu: {str(e)}'
        print(error_message)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)