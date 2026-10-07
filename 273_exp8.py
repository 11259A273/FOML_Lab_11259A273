import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load the Heart dataset
data = pd.read_csv("Heart.csv")

# Remove unwanted index column if present
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# Separate input and output
X = data.drop("AHD", axis=1)
y = data["AHD"]

# Convert output Yes/No into 1/0
y = y.map({"Yes": 1, "No": 0})

# Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)

# Fill missing values
X = X.fillna(X.median(numeric_only=True))

# Convert any remaining missing categorical values to 0
X = X.fillna(0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

# Create Gradient Boosting model
model = GradientBoostingClassifier(random_state=1)

# Train the model
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", round(accuracy, 3))

# Take one patient from test data
new_patient = X_test.iloc[[0]]

# Predict
result = model.predict(new_patient)[0]

print("Heart disease? (1=yes, 0=no):", result)