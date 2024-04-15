from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from joblib import load
import pandas as pd
def rateModal():
    model = load('app/models/gradient_boosting_model.joblib')
    # Định nghĩa mô hình đã có
    # Ví dụ: model = RandomForestClassifier()
    data = pd.read_excel('app/services/data_final.xlsx')

    # Loại bỏ các mẫu có giá trị NaN từ tập dữ liệu huấn luyện
    data = data.dropna()
    features = ['Type', 'Area', 'Bedrooms', 'Bathrooms']
    X = data[features]
    y=data['Price']
    # Tạo đối tượng KFold với k=5
    kf = KFold(n_splits=5)

    # Sử dụng cross_val_score để thực hiện k-fold cross-validation
    scores = cross_val_score(model, X, y, cv=kf)

    # In ra kết quả
    print("Accuracy:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())
    return {scores, scores.mean(), scores.std()}