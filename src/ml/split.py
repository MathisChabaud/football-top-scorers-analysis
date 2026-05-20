from sklearn.model_selection import train_test_split


def classic_train_test_split(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets using a random split with stratification.
    """

    features = df.drop(columns=['scored'])
    target = df['scored']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    return X_train, X_test, y_train, y_test


def train_test_split_by_date(df_features, train_ratio=0.8):
    """
    Splits the DataFrame into training and testing sets based on the date :
    the training set consists of the earliest data, and the testing set consists of the most recent data.
    """

    df_features = df_features.sort_values("date")
    split_index = int(len(df_features) * train_ratio)
    train_df = df_features.iloc[:split_index]
    test_df = df_features.iloc[split_index:]

    X_train = train_df.drop(columns=["scored"])
    y_train = train_df["scored"]
    X_test = test_df.drop(columns=["scored"])
    y_test = test_df["scored"]

    return X_train, X_test, y_train, y_test
