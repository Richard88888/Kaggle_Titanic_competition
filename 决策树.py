import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# 第一步：数据探索
print(train_data.info())  # 了解数据表的基本情况：行数、列数、每列的数据类型、数据完整度
print('-'*30)
print(train_data.describe())  # 了解数据表的统计情况：总数、平均值、标准差、最小值、最大值等
print('-'*30)
print(train_data.describe(include=['O']))  # 查看字符串类型（非数字）的整体情况
print('-'*30)
print(train_data.head())  # 查看前几行数据（默认是前5行）
print('-'*30)
print(train_data.tail())  # 查看后几行数据（默认是最后5行）
print('-'*30)
# 第二步：数据清洗
# 使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print('-'*30)
print(train_data['Embarked'].value_counts())
# 使用登陆最多的港口来填充港口的nan值
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)
# 第三步：特征选择
# PassengerId 为乘客编号，对分类没有作用，可以放弃；Name 为乘客姓名，对分类没有作用，可以放弃；Cabin 字段缺失值太多，可以放弃；Ticket 字段为船票号码，杂乱无章且无规律，可以放弃。
# 其余的字段包括：Pclass、Sex、Age、SibSp、Parch 和 Fare，这些属性分别表示了乘客的船票等级、性别、年龄、亲戚数量以及船票价格，可能会和乘客的生存预测分类有关系。
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
# 将特征值中字符串表示为0.1表示
dvec = DictVectorizer(sparse=False)
# fit_transform() 将特征向量转化为特征值矩阵
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
# 查看dvec转化后的特征属性
print(dvec.feature_names_)
# 第四步：构建决策树模型
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features,train_labels)
# 第五步：模型预测
test_features = dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features).astype(np.int64)
print(pred_labels)
id = test_data['PassengerId']
sub = {'PassengerId': id, 'Survived': pred_labels}
submission = pd.DataFrame(sub)
submission.to_csv("submission.csv", index=False)