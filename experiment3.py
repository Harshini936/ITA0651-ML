from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X,y = load_iris(return_X_y=True)

m = DecisionTreeClassifier(criterion="entropy").fit(X,y)

print("Prediction:", m.predict([[5.1,3.5,1.4,0.2]]))
