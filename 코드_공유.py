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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# GridSearch를 활용한 하이퍼파라미터 튜닝
param_grid = {
    "criterion" : ['gini', 'entropy'],
    "max_depth" : [2, 3, 4, 5],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4]
}

# HPO
clf_grid = DecisionTreeClassifier(random_state=42)

# core
grid_search = GridSearchCV(clf_grid, param_grid, cv =5)

# 하이퍼파라미터를 찾고, fitting 수행
grid_search.fit(X_train,y_train)

# accuracy를 기준으로 모델 정확도 평가
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_grid = best_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

# 특성 Importacne
importances = best_model.feature_importances_

print(f"Best Hyper-parmeter {best_params}")
print(f"Best Score {accuracy_grid}")

# Best model의 Feature Importance를  시각화
plt.figure(figsize = (20,6))

# 막대 그래프 생성
plt.bar(range(len(importances)), importances, width=0.3)
plt.xlabel('Feature')
plt.ylabel('importances')
plt.title('Feature Importance')
plt.xticks(range(len(importances)), X.columns, rotation = 45)
plt.show()




####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

