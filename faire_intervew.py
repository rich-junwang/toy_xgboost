import pandas as pd
from xgboost import plot_importance
from matplotlib import pyplot
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


data = pd.read_csv("./faire-ml-rank-small.csv")

# data = data.head(100)
# data.to_csv("faire-ml-rank-tiny.csv")

print("original data shape", data.shape)
y = data["has_product_click"][:]

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
features = data.select_dtypes(include=numerics)
features.drop('has_product_click', inplace=True, axis=1)
print("selected feature data shape", features.shape)

columns = features.columns.values.tolist()
print("current feature being used: ", columns)


data_dmatrix = xgb.DMatrix(data=features, label=y)


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBClassifier(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# xg_reg.get_score(importance_type='gain')
print("accuracy: %f" % (accuracy))
print(classification_report(y_test, preds))


plot_importance(xg_reg)
pyplot.show()
