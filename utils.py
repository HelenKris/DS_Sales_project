import pandas as pd

def get_data(data):
    """Extracts features and labels from the given dataset.

    Args:
        data (DataFrame): The input dataset containing features and labels.

    Returns:
        tuple: A tuple containing features (X_train) and labels (y_train).
    """
    # Find the column with the maximum index
    y_col_name = max([col for col in data.columns if isinstance(col, int)])
    X_train = data.drop([y_col_name], axis=1)
    y_train = data[y_col_name]
    # Rename the lags to x_train
    rename_dict = {
        col: f"lag {i+1}"
        for i, col in enumerate(
            [col for col in X_train.columns if isinstance(col, int)][::-1]
        )
    }
    X_train.rename(columns=rename_dict, inplace=True)
    X_train = X_train.round().astype(int)
    y_train.columns = ["y"]
    return X_train, y_train

def downcast(df, verbose=False):
    if not isinstance(df, pd.DataFrame):
        return df  # Skip if not a DataFrame
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype_name = df[col].dtype.name
        if dtype_name == "object":
            pass
        elif dtype_name == "bool":
            df[col] = df[col].astype("int8")
        elif dtype_name.startswith("int") or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print("{:.1f}% compressed".format(100 * (start_mem - end_mem) / start_mem))
    return df
