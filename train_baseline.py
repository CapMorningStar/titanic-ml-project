import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1) Load data
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# 2) Define target and features
y = train_df["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_df[features]

# 3) Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Preprocessing
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5) Model
model = LogisticRegression(max_iter=1000)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# 6) Train
clf.fit(X_train, y_train)

# 7) Evaluate
preds = clf.predict(X_val)
acc = accuracy_score(y_val, preds)

print("Validation Accuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_val, preds))
print("\nClassification Report:\n", classification_report(y_val, preds))
