# from src.features.build_features import FeatureEngineering
# from utils import get_data, downcast
import get_data, downcast
import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import click


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepaths", type=click.Path(), nargs=5)
def prepare_datasets(input_filepath, output_filepaths):

    prepared_data = pd.read_csv(input_filepath)
    test_block_num = prepared_data["date_block_num"].max() + 1

    label_encoder = LabelEncoder()

    # Пустые списки для хранения данных
    X_train_list = []
    y_train_list = []

    # In the cycle we go through a window of 12 months
    # with a step of 1 month and form parts of the dataset for subsequent cocatenation
    for end_block_num in range(test_block_num, -1, -1):
        if end_block_num - 12 >= 0:
            start_block_num = end_block_num - 12
            processor = FeatureEngineering(
                start_block_num=start_block_num,
                end_block_num=end_block_num,
                label_encoder=label_encoder,
                # clip_threshold=clip_threshold,
            )
            train_data = processor.get_features(prepared_data)
            X_train, y_train = get_data(train_data)
            X_train_list.append(downcast(X_train))
            y_train_list.append(y_train)

    # We create a test dataset where end_block_num=test_block_num and  start_block_num=end_block_num - 12
    X_test = X_train_list[0]
    # y_test = y_train_list[0]

    # We create a test dataset where end_block_num=(test_block_num - 1)  and  start_block_num=(end_block_num - 12 -1)
    X_val = pd.concat(
        [X_train_list[1], X_train_list[len(X_train_list) - 1]], ignore_index=True
    )
    y_val = pd.concat(
        [y_train_list[1], y_train_list[len(y_train_list) - 1]], ignore_index=True
    )

    # Form a training dataset by concatenating all subsequent parts of the dataset
    X_train = X_train_list[2]
    y_train = y_train_list[2]

    for i in range(3, len(X_train_list)):
        X_train = pd.concat([X_train, X_train_list[i]], ignore_index=True)
        y_train = pd.concat([y_train, y_train_list[i]], ignore_index=True)

    # Save all the  datasets to the specified path
    X_train.to_csv(output_filepaths[0], index=False)
    y_train.to_csv(output_filepaths[1], index=False)
    X_val.to_csv(output_filepaths[2], index=False)
    y_val.to_csv(output_filepaths[3], index=False)
    X_test.to_csv(output_filepaths[4], index=False)


if __name__ == "__main__":
    prepare_datasets()
