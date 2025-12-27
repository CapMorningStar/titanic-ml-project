import pandas as pd

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)

print("\nTRAIN head:")
print(train_df.head())

print("\nTRAIN info:")
print(train_df.info())

print("\nMissing values in TRAIN:")
print(train_df.isna().sum().sort_values(ascending=False).head(10))

y = train_df["Survived"]
print("\nTarget (Survived) distribution:")
print(y.value_counts())
