from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


boston = load_boston()
print(boston.data.shape)

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
# print(data.head())
# data.info()

X, y = data.iloc[:, :-1], data.iloc[:, -1]

data_dmatrix = xgb.DMatrix(data=X,label=y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(type(X_train))
exit(1)
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
