from joblib import load
import pandas as pd
import numpy as np

def predict_price(type_value, area_value, bedrooms_value, bathrooms_value, location_value):
    # Load mô hình Gradient Boosting từ tệp
    loaded_model = load('app/models/gradient_boosting_model.joblib')

    # Load danh sách các vị trí
    locations = ['Cẩm Lệ', 'Hải Châu', 'Hòa Vang', 'Liên Chiểu', 'Ngũ Hành Sơn', 'Sơn Trà', 'Thanh Khê']

    # Tạo DataFrame từ các giá trị đầu vào
    new_data = pd.DataFrame({
        'Type': [type_value],
        'Area': [area_value],
        'Bedrooms': [bedrooms_value],
        'Bathrooms': [bathrooms_value]
    })

    # Tạo one-hot encoding cho vị trí nhập từ người dùng
    for loc in locations:
        new_data[f'Location_{loc}'] = 1 if loc == location_value else 0

    # Dự đoán giá căn hộ mới
    predicted_price = loaded_model.predict(new_data)

    # In ra giá dự đoán của căn hộ mới
    return "{:,.2f}".format(round(predicted_price[0], 0))

def propose(type_value, area_value, bedrooms_value, bathrooms_value, location_value, price_value):
    # Đọc dữ liệu từ tệp Excel vào DataFrame
    data = pd.read_excel("app/services/data_final.xlsx")
    data = data.dropna()

    # Tạo các cột tương ứng cho vị trí
    locations = ['Cẩm Lệ', 'Hải Châu', 'Hòa Vang', 'Liên Chiểu', 'Ngũ Hành Sơn', 'Sơn Trà', 'Thanh Khê']
    for loc in locations:
        data[f'Location_{loc}'] = (data['Location'] == loc).astype(int)

    # Chọn các đặc trưng cần thiết và chuyển đổi chúng thành vector đặc trưng
    features = data[['Type', 'Area', 'Bedrooms', 'Bathrooms', 'Price'] + [f'Location_{loc}' for loc in locations]]
    print(features)
    
    # Tạo DataFrame từ các giá trị đầu vào
    new_data = pd.DataFrame({
        'Type': [type_value],
        'Area': [area_value],
        'Bedrooms': [bedrooms_value],
        'Bathrooms': [bathrooms_value],
        'Price' : [price_value]
    })

    # Tạo one-hot encoding cho vị trí nhập từ người dùng
    for loc in locations:
        new_data[f'Location_{loc}'] = 1 if loc == location_value else 0

    # Tính toán khoảng cách giữa vector đặc trưng của căn hộ mà người dùng nhập và tất cả các căn hộ trong dữ liệu
    distances = np.linalg.norm(features.values - new_data.values, axis=1)

    # Lấy index của các căn hộ có khoảng cách gần nhất
    closest_indices = np.argsort(distances)

    # In ra các căn hộ gần tương tự nhất
    closest_apartments = data.iloc[closest_indices[:5]]  # Lấy 5 căn hộ gần nhất
    return closest_apartments.to_dict(orient='records')