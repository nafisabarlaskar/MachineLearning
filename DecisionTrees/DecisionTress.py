from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
pred = clf.predict([[2., 2.]])
print(pred)
pred_prob = clf.predict_proba([[2., 2.]])
print(pred_prob)

iris = load_iris()
irs = tree.DecisionTreeClassifier()
irs = irs.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

