from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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

new_tumour = [X_test[0]]

result = forest.predict(new_tumour)[0]

print("Diagnosis:", names[result])

importances = forest.feature_importances_

top = sorted(
    zip(importances, data.feature_names),
    reverse=True
)[:5]

vals = [x[0] for x in top]
labels = [x[1] for x in top]

plt.bar(labels, vals)
plt.xlabel("Importance")
plt.title("Top 5 Most Important Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()