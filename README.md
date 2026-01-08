# 1. Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load the dataset
df = pd.read_csv("customer_churn.csv")

# 3. Display basic info
print(df.head())
print(df.info())

# 4. Handle missing values (simple approach)
df = df.dropna()

# 5. Encode categorical variables
label_encoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# 6. Separate features (X) and target (y)
X = df.drop("Churn?", axis=1)   # change if your target column name is different
y = df["Churn?"]

# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
