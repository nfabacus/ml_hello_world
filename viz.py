import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

## https://en.wikipedia.org/wiki/Iris_flower_data_set#Data_set

iris = load_iris()
## iris object is like { 'data': [6.8, 3.2, 5.9, 2.3], ...., 'target': [0,0, 1,....] }

print('####### feature names and corresponding values for the features')
print(iris.feature_names)
print(iris.data[0])
print('####### target names and corresponding label value')
print(iris.target_names)
print(iris.target[0])

# for i in range(len(iris.target)):
#     print("Example {}: features {}, label {}".format(i, iris.data[i], iris.target[i]))

test_index = [0, 50, 100]

# training data (all data apart from the removed testing data (3 items))
train_target = np.delete(iris.target, test_index)
print('train target>>>>', train_target)

train_data = np.delete(iris.data, test_index, axis=0)
print('train_data>>>', train_data)

# testing data (testing data from the original data set - 3 items).
test_target = iris.target[test_index]
print('test_target>>>>', test_target)
test_data = iris.data[test_index]
print('test_data>>>>', test_data)


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# checking below if the original target (labels) for the three items match will the prediction for the test data.
print(test_target)
print(clf.predict(test_data))






