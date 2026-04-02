import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def inspect_data(df):
    """Print basic dataset information."""
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["HeartDisease"].value_counts())

    print("\nZero values that are suspicious:")
    print("RestingBP zeros:", (df["RestingBP"] == 0).sum())
    print("Cholesterol zeros:", (df["Cholesterol"] == 0).sum())


def split_features_target(df):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    return X, y


def encode_features(X, categorical_columns):
    return pd.get_dummies(X, columns=categorical_columns, drop_first=True)


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_data(X_train, X_test, numeric_columns):
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train, X_test


def save_processed_data(X_train, X_test, y_train, y_test):
    X_train.to_csv("X_train_processed.csv", index=False)
    X_test.to_csv("X_test_processed.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)


def main():
    df = load_data("heart.csv")

    print("Dataset audit")
    inspect_data(df)

    categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numeric_columns = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    X, y = split_features_target(df)
    X_encoded = encode_features(X, categorical_columns)

    print("\nEncoded feature columns:")
    print(X_encoded.columns.tolist())

    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    X_train, X_test = scale_data(X_train, X_test, numeric_columns)

    print("\nProcessed data")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    save_processed_data(X_train, X_test, y_train, y_test)
    print("\nProcessed files saved")


if __name__ == "__main__":
    main()