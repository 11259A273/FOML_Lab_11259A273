from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=0
)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Decision Tree accuracy:", round(tree.score(X_test, y_test), 3))
print("Random Forest accuracy:", round(forest.score(X_test, y_test), 3))
names = data.target_names
new_tumour =[X_test[0]]
result = forest.predict(new_tumour)[0]
print("Diagnosis:", names[result])