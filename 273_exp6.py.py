import pandas as pd
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv(r"C:\Users\Admin\PycharmProjects\PythonProject56\Mall_Customers (1).csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

model = AgglomerativeClustering(n_clusters=5)

data["Group"] = model.fit_predict(X)

print(data.groupby("Group")[["Annual Income (k$)", "Spending Score (1-100)"]].mean())