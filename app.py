from flask import Flask, request, jsonify
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
    'Neural Network (MLP)': MLPRegressor(max_iter=1000)
}

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Çalışıyor'})

# Veri ve analiz işlemlerini birleştirip tek endpoint'e çeken kod
@app.route('/analyze', methods=['POST'])
def analyze_data():
    file = request.files.get('file')
    dependent_variable = request.form.get('dependent_variable')

    if not file or not dependent_variable:
        return jsonify({'error': 'Dosya veya bağımlı değişken eksik.'}), 400

    try:
        # Dosya işlemlerini alıp veri çerçevesine dönüştürme
        df = pd.read_excel(file)
        if dependent_variable not in df.columns:
            return jsonify({'error': f'Bağımlı değişken {dependent_variable} veri setinde bulunamadı.'}), 400

        # Kategorik değişkenleri dönüştürme ve modelleme için veri hazırlığı
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
                    'r2_score': round(r2, 4)
                }
            except Exception as e:
                results[model_name] = {'error': str(e)}

        return jsonify(results)
    except Exception as e:
        error_message = f'Analiz sırasında bir hata oluştu: {str(e)}'
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
