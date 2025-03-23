# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt


wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
# 데이터프레임 생성
df = pd.DataFrame(data=wine.data, columns = wine.feature_names)
df['target'] = wine.target

# feature와 target의 데이터 분리
X = df.drop('target', axis=1)
y = df['target']

# 학습데이터와 테스트 데이터로 분리(test size 0.2, random_state 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# 학습데이터 구성 및 전처리
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 레이블링 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# XGboost 모델 생성 및 학습
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train_encoded)


# 예측 및 레이블 디코딩
xgb_y_pred_encoded = xgb_model.predict(X_test)

# 하이퍼파라미터 범위지정
params = {
    "max_depth" : [3, 5, 7, 9, 15],
    "learning_rate" : [0.1, 0.01, 0.001],
    "n_estimators": [50, 100, 200, 300]
}

# 하이퍼파라미터 최적화 
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train_encoded)

# 최적의 하이퍼파라미터의 학습
best_params_xgb = grid_search_xgb.best_params_
best_model_xgb = grid_search_xgb.best_estimator_

#테스트 데이터에 대한 예측
y_pred_encoded = best_model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test_encoded, y_pred_encoded)

print(f"Best Hyper-parmeter {best_params_xgb}")
print(f"Best Score {accuracy_xgb}")

# Feature Importance 시각화
importances = best_model_xgb.feature_importances_

plt.figure(figsize= (20,12))
#막대그래프 생성
plt.bar(range(len(importances)), importances, width= 0.3)
plt.xlabel('Feature')
plt.ylabel('importance')
plt.title('Feature Importance')
plt.xticks(range(len(importances)), X.columns, rotation =45)
plt.show()