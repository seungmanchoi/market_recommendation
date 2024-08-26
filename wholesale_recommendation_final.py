import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import random

# 1. 모델과 관련된 객체 불러오기
print("모델과 관련 객체 로딩 중...")
model = joblib.load('wholesale_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
mlb = joblib.load('category_mlb.pkl')
features = joblib.load('features_list.pkl')

# 2. 데이터 불러오기
print("데이터 로딩 중...")
data = pd.read_csv('data.csv')

# 3. 데이터 전처리 및 피처 변환
print("데이터 전처리 중...")
data_sampled = data.reset_index(drop=True)

# 데이터 타입 최적화
data_sampled['product_clicks'] = data_sampled['product_clicks'].astype('int32')
data_sampled['market_clicks'] = data_sampled['market_clicks'].astype('int32')
data_sampled['total_order_price'] = data_sampled['total_order_price'].astype('float32')

# 카테고리 처리
data_sampled['category'] = data_sampled['category'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
category_encoded = mlb.transform(data_sampled['category'])

# 소매 카테고리와 도매 카테고리 처리
data_sampled['wholesale_category'] = pd.factorize(data_sampled['wholesale_category'])[0]

# retail_id에 대해서만 One-hot 인코딩
data_sampled = pd.get_dummies(data_sampled, columns=['retail_id'], drop_first=False)

# 카테고리 원-핫 인코딩 추가
category_df = pd.DataFrame(category_encoded, columns=mlb.classes_)
data_sampled = pd.concat([data_sampled, category_df], axis=1)

# 필요 없는 열 제거
data_sampled = data_sampled.drop(['category'], axis=1)

# wholesale_id 열을 포함시키기 위해 원본 데이터에서 추출
data_sampled['wholesale_id'] = data['wholesale_id']

# 4. 피처 일관성 유지 및 누락된 피처 처리
# print("피처 일관성 유지 및 누락된 피처 처리 중...")
missing_features = {feature: 0 for feature in features if feature not in data_sampled.columns}
missing_features_df = pd.DataFrame(missing_features, index=data_sampled.index)
data_sampled = pd.concat([data_sampled, missing_features_df], axis=1)

# 피처 이름 일관성 유지: 모델 학습 시 사용된 피처 이름에 맞춰 데이터 재정렬
data_sampled = data_sampled.reindex(columns=features + ['order_count', 'wholesale_id'], fill_value=0)

# 5. 소매 전체에 대해 AUC 계산 및 추천
print("소매 전체에 대한 AUC 계산 및 추천 시작...")
auc_값들 = []
추천결과 = {}
가능한_retail_columns = []

for column in data_sampled.columns:
    if column.startswith('retail_id_'):
        retail_데이터 = data_sampled[data_sampled[column] == 1]

        if not retail_데이터.empty:
            # wholesale_id 열을 별도로 저장하고 이후 분석에서 제외
            wholesale_ids = retail_데이터['wholesale_id']
            X = retail_데이터.drop(['order_count', 'wholesale_id'], axis=1)

            # 타겟 정의
            y = (retail_데이터['order_count'] > 0).astype(int)

            # NaN 값을 0으로 대체
            X.fillna(0, inplace=True)

            # 스케일링
            X_scaled = scaler.transform(X)

            # 모델 예측
            y_scores = model.decision_function(X_scaled)

            # ROC 커브 및 AUC 계산
            if y.nunique() == 2:
                fpr, tpr, _ = roc_curve(y, y_scores)
                roc_auc = auc(fpr, tpr)
                auc_값들.append(roc_auc)
                가능한_retail_columns.append(column)

            # 상위 5개의 도매 아이디 추천
            추천_도매 = wholesale_ids.iloc[np.argsort(-y_scores)].unique()[:5]
            추천결과[column] = 추천_도매

# 6. 소매 100개 랜덤 추출 및 도매 추천
if len(가능한_retail_columns) >= 100:
    selected_retail_columns = random.sample(가능한_retail_columns, 100)
else:
    selected_retail_columns = 가능한_retail_columns

print(f"\n최종 선택된 100개 소매에 대한 도매 추천:")
for retail_col in selected_retail_columns:
    도매_추천 = 추천결과.get(retail_col, [])
    print(f"{retail_col}: 추천 도매 아이디 - {도매_추천}")

# 전체 평균 AUC 계산
if auc_값들:
    평균_auc = np.mean(auc_값들)
    print(f"\n평균 AUC: {평균_auc:.4f}")
else:
    print("평가 가능한 소매 ID가 없습니다.")

"""
target - 주문을 한다.(할 것이다.), 실제 일촌여부는 예측의 상관관계가 낮을 수 있음.

머신러닝 모델 (SGDClassifier)
- 입력 피처의 가중치(weight)와 편향(bias)을 학습하여 입력 데이터가 어떤 클래스에 속하는지를 예측

추천 시스템 (Recommendation)
예측 기반 추천: 모델이 예측한 점수를 기반으로 각 소매에게 추천할 도매 리스트를 제공. 점수가 높을수록 해당 도매에서 구매할 가능성이 크다고 예측.
추천 방식: 각 소매에 대해 가장 높은 점수를 받은 도매 ID를 추천. 이 점수는 모델이 도출한 예측 값. 
(점수 도출은 SGDClassifier가 수행하고 구매와 큰 상관관계가 있다고 판단하는 피처에 대해 높은 가중치를 부여하고 낮다고 판단하는거에 대해 낮은 가중치를 부여함)

매장클릭수, 매장의 상품 클릭수, 주문수, 주문금액, 카테고리

1. True Positive (TP)
정의: 모델이 Positive(긍정적)으로 예측한 값이 실제로도 Positive인 경우.
적용 예시: 모델이 "구매할 것이다"라고 예측했는데, 실제로도 그 고객이 구매를 한 경우.

2. False Positive (FP)
정의: 모델이 Positive로 예측했으나 실제로는 Negative(부정적)인 경우.
적용 예시: 모델이 "구매할 것이다"라고 예측했는데 실제로는 그 고객이 구매하지 않은 경우.

3. True Negative (TN)
정의: 모델이 Negative로 예측한 값이 실제로도 Negative인 경우.
적용 예시: 모델이 "구매하지 않을 것이다"라고 예측했는데 실제로도 그 고객이 구매하지 않은 경우.

4. False Negative (FN)
정의: 모델이 Negative로 예측했으나 실제로는 Positive인 경우.
적용 예시: 모델이 "구매하지 않을 것이다"라고 예측했는데 실제로는 그 고객이 구매한 경우.

구매 예측 모델에서의 TP와 FP:
True Positive (TP): 모델이 고객이 "구매할 것이다"라고 예측했고 실제로 그 고객이 구매한 경우.
False Positive (FP): 모델이 고객이 "구매할 것이다"라고 예측했지만 실제로 그 고객이 구매하지 않은 경우.
"""
