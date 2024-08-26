import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc

# 모델 초기화
model = SGDClassifier(loss='log', random_state=42, max_iter=1000, tol=1e-3)

# 스케일러와 멀티라벨 바이너라이저 초기화
scaler = StandardScaler()
mlb = MultiLabelBinarizer()

# CSV 파일 경로
file_path = 'data.csv'

# 데이터 배치 크기 설정
batch_size = 10000

# 학습 데이터를 처리하는 루프
first_batch = True
batch_number = 0
feature_names = None

for chunk in pd.read_csv(file_path, chunksize=batch_size):
    batch_number += 1
    print(f"\n[INFO] Processing batch {batch_number}...")

    # 데이터 타입 최적화
    chunk['product_clicks'] = chunk['product_clicks'].astype('int32')
    chunk['market_clicks'] = chunk['market_clicks'].astype('int32')
    chunk['total_order_price'] = chunk['total_order_price'].astype('float32')

    # 카테고리 처리
    chunk['category'] = chunk['category'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # 처음 배치인 경우, 멀티라벨 바이너라이저 학습
    if first_batch:
        category_encoded = mlb.fit_transform(chunk['category'])
        first_batch = False
        print("[INFO] MultiLabelBinarizer fitted on first batch.")
    else:
        category_encoded = mlb.transform(chunk['category'])

    # 소매 카테고리와 도매 카테고리 처리
    chunk['wholesale_category'] = pd.factorize(chunk['wholesale_category'])[0]

    # One-hot encoding for retail_id and wholesale_id
    chunk = pd.get_dummies(chunk, columns=['retail_id', 'wholesale_id'], drop_first=True)

    # 카테고리 원-핫 인코딩 추가
    category_df = pd.DataFrame(category_encoded, columns=mlb.classes_)
    chunk = pd.concat([chunk, category_df], axis=1)

    # 필요 없는 열 제거
    chunk = chunk.drop(['category'], axis=1)

    # 타겟 변수 및 특성 정의
    X = chunk.drop(['order_count'], axis=1)
    y = (chunk['order_count'] > 0).astype(int)

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # NaN 값을 0으로 대체 (학습 및 테스트 데이터 모두)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # 첫 번째 배치에서 피처 이름 저장 및 스케일링
    if batch_number == 1:
        feature_names = X_train.columns  # 모든 피처 이름을 저장
        X_train_scaled = scaler.fit_transform(X_train)
        print("[INFO] StandardScaler fitted on first batch.")
    else:
        # 이후 배치에서 피처 이름 일관성 유지 및 스케일링
        X_train = X_train.reindex(columns=feature_names, fill_value=0)
        X_train_scaled = scaler.transform(X_train)

    # 모델 배치 학습
    model.partial_fit(X_train_scaled, y_train, classes=np.unique(y_train))
    print(f"[INFO] Batch {batch_number} processed and model updated.")

    # 메모리 정리
    del X, y, chunk, X_train, y_train, X_train_scaled
    gc.collect()

print("모델 학습 완료")

# 피처 이름 저장 (전체 피처 이름을 저장)
joblib.dump(feature_names.tolist(), 'features_list.pkl')

# 테스트 데이터 스케일링 및 평가
X_test = X_test.reindex(columns=feature_names, fill_value=0)
X_test_scaled = scaler.transform(X_test)

# 예측 및 평가
y_scores = model.decision_function(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
print(f"테스트 데이터 AUC: {roc_auc:.4f}")

# 모델과 관련된 객체 저장
joblib.dump(model, 'wholesale_recommendation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(mlb, 'category_mlb.pkl')

print("모델과 관련 객체들이 저장되었습니다.")

# ROC 커브 시각화 및 이미지 저장
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve_plot.png')
plt.show()

print("ROC 커브 이미지가 'roc_curve_plot.png' 파일로 저장되었습니다.")
