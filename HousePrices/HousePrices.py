import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# select features
selected_features = ['Foundation', 'Heating', 'Electrical', 'SaleType', 'SaleCondition', 'GarageArea','YearRemodAdd','YearBuilt','1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'BsmtUnfSF', 'CentralAir']

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['SalePrice']

# 补充缺失特征值
X_train['Electrical'].fillna('SBrkr', inplace=True)
X_train['SaleType'].fillna('WD', inplace=True)
X_train['GarageArea'].fillna(X_train['GarageArea'].mean(), inplace=True)
X_train['TotalBsmtSF'].fillna(X_train['TotalBsmtSF'].mean(), inplace=True)
X_train['BsmtUnfSF'].fillna(X_train['BsmtUnfSF'].mean(), inplace=True)
X_test['Electrical'].fillna('SBrkr', inplace=True)
X_test['SaleType'].fillna('WD', inplace=True)
X_test['GarageArea'].fillna(X_test['GarageArea'].mean(), inplace=True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(), inplace=True)
X_test['BsmtUnfSF'].fillna(X_test['BsmtUnfSF'].mean(), inplace=True)

print(X_train.info())
print(X_test.info())

# 采用 DictVectorizer 进行特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 使用随机森林回归模型进行回归预测
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)

# 输出结果
regressor_submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_predict})
regressor_submission.to_csv('submission.csv', index=False)
