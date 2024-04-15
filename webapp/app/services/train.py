import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from tabulate import tabulate
import numpy as np

# Đọc dữ liệu từ file Excel
data = pd.read_excel('app/services/data_final.xlsx')

# Loại bỏ các mẫu có giá trị NaN từ tập dữ liệu huấn luyện
data = data.dropna()

locations = ['Cẩm Lệ', 'Hải Châu', 'Hòa Vang', 'Liên Chiểu', 'Ngũ Hành Sơn', 'Sơn Trà', 'Thanh Khê']

# Khởi tạo one-hot encoding cho tất cả các vị trí
for loc in locations:
    data[f'Location_{loc}'] = (data['Location'] == loc).astype(int)

# Chọn các đặc trưng (features) và biến mục tiêu (target)
features = ['Type', 'Area', 'Bedrooms', 'Bathrooms'] + [f'Location_{loc}' for loc in locations]
target = 'Price'

# Tách dữ liệu thành features và target
X = data[features]
y = data[target]


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Danh sách các mô hình để thử nghiệm
models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "random_forest": RandomForestRegressor()
}

# Kết quả của các mô hình
results = []
best_model = None
best_mse_rsquared_rmse = None
best_score = float('inf')
# Huấn luyện từng mô hình và đánh giá
for name, model in models.items():
    print(f"Check is best model? : {name}...")
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tính MSE, R-squared và RMSE
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    combined_score = mse + (1 - r_squared) + rmse  # Kết hợp các chỉ số đánh giá
    if combined_score < best_score:
        best_score = combined_score
        best_mse_rsquared_rmse = (mse, r_squared, rmse)
        best_model = (name, model)
        
    results.append([name, mse, r_squared, rmse])

# Hiển thị kết quả tốt nhất của mỗi mô hình
print(tabulate(results, headers=["Model", "MSE", "R-squared", "RMSE"], tablefmt="grid"))
print("Best Model:", best_model[0])
best_mse, best_r_squared, best_rmse = best_mse_rsquared_rmse
print("MSE:", best_mse)
print("R-squared:", best_r_squared)
print("RMSE:", best_rmse)

best_model_name, best_model_instance = best_model

# Xác định các tham số bạn muốn thử nghiệm
param_grid = {
    "linear_regression": {},
    "decision_tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "gradient_boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 10]
    },
    "random_forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [5, 10, 15]
    }
}

# Tạo một đối tượng GridSearchCV
grid_search = GridSearchCV(best_model_instance, param_grid[best_model_name], cv=5, scoring='neg_mean_squared_error')

# Thực hiện tìm kiếm trên lưới
grid_search.fit(X_train, y_train)

# In ra các tham số tốt nhất
print("Best Parameters:", grid_search.best_params_)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Lưu mô hình Gradient Boosting ra tệp
dump(best_model, f'{best_model_name}_model.joblib')