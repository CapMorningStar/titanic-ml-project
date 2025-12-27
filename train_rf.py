import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Family features
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        # Title from Name
        X["Title"] = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        X["Title"] = X["Title"].replace(
            {
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
                "Lady": "Rare",
                "Countess": "Rare",
                "Capt": "Rare",
                "Col": "Rare",
                "Don": "Rare",
                "Dr": "Rare",
                "Major": "Rare",
                "Rev": "Rare",
                "Sir": "Rare",
                "Jonkheer": "Rare",
                "Dona": "Rare",
            }
        )
        X["Title"] = X["Title"].fillna("Unknown")

        return X


# 1) Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 2) Split X / y
y = train_df["Survived"]
X = train_df.drop(columns=["Survived"])

# 3) Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Columns after feature engineering
numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
categorical_features = ["Pclass", "Sex", "Embarked", "Title"]

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
    ],
    remainder="drop"
)

# 5) Model
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    max_depth=6,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("fe", FeatureEngineer()),
    ("preprocess", preprocess),
    ("model", rf)
])

# 6) Train + evaluate
clf.fit(X_train, y_train)
preds = clf.predict(X_val)

acc = accuracy_score(y_val, preds)
print("Validation Accuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_val, preds))
print("\nClassification Report:\n", classification_report(y_val, preds))

# 7) Train on full data and create Kaggle submission
clf.fit(X, y)
test_preds = clf.predict(test_df)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nSaved: submission.csv")
