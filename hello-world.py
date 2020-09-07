from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]] # e.g. [weight e.g 140gram, texture e.g. 0 for "bumpy", 1 for "smooth" ]
labels = [0, 0, 1, 1]  # 0 for apple, 1 for orange
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)  # fit is to find patterns in data.
print(clf.predict([[150, 0]]))  # e.g. predit for an item which weighs 150g and with bumpy surface.
