# from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# import sys
# sys.path.append('D:/Innowise/DS project')
# import config

# prepared_data = pd.read_csv(config.prepared_data_path)


class FeatureEngineering:
    """A class for generating features for a given dataset.
    Args:
        start_block_num (int): The starting block number for feature generation.
        end_block_num (int): The ending block number for feature generation.
        label_encoder (object): The label encoder object for encoding categorical features.
        clip_threshold (int): The threshold for clipping item count.
    Attributes:
        start_block_num (int): The starting block number for feature generation.
        end_block_num (int): The ending block number for feature generation.
        label_encoder (object): The label encoder object for encoding categorical features.
        clip_threshold (int): The threshold for clipping item count.
    Methods:
        _get_price_dynamics(data): Generates price dynamics features based on the part of the dataset.
        _lag_item_count(data): Generates lagged item count features based on the part of the dataset.
        _get_mean_features(data): Generates mean features based on the part of the dataset.
        _get_other_features(data): Generates other features based on the part of the dataset.
        _label_cat_features(data): Labels categorical features based on the part of the dataset.
        get_features(data): Combines all feature generation methods and returns the final part of the dataset with features.
    """

    def __init__(self, start_block_num, end_block_num, label_encoder):
        """Initializes FeatureEngineering class with the provided parameters."""

        self.start_block_num = start_block_num
        self.end_block_num = end_block_num
        self.label_encoder = label_encoder

    def _get_price_dynamics(self, data):
        """Generates price dynamics features based on the part of the dataset."""

        df_pivot = data.pivot_table(
            index=[
                "shop_id",
                "item_category_id",
                "item_id",
                "item_category_name",
                "shop_name",
            ],
            columns="date_block_num",
            values="item_price",
            aggfunc="mean",
        ).fillna(0.0)
        df_pivot = df_pivot.reset_index()
        str_columns = [col for col in df_pivot.columns if isinstance(col, str)]
        int_columns = [col for col in df_pivot.columns if isinstance(col, int)]
        df_pivot = df_pivot.groupby(str_columns)[int_columns].sum().reset_index()
        price_dynamics = df_pivot.loc[:, self.start_block_num : self.end_block_num - 1]
        rename_dict = {
            col: f"Price {i + 1}"
            for i, col in enumerate([col for col in price_dynamics.columns][::-1])
        }
        price_dynamics.rename(columns=rename_dict, inplace=True)
        data = pd.concat([df_pivot[str_columns], price_dynamics], axis=1)
        return data

    def _lag_item_count(self, data):
        """Generates lagged item count features based on the part of the dataset."""

        data["item_cnt"] = data["item_cnt"].clip(0, 20)
        df_pivot = data.pivot_table(
            index=[
                "shop_id",
                "item_category_id",
                "item_id",
                "item_category_name",
                "shop_name",
            ],
            columns="date_block_num",
            values="item_cnt",
            aggfunc="sum",
        ).fillna(0.0)
        df_pivot = df_pivot.reset_index()
        str_columns = [col for col in df_pivot.columns if isinstance(col, str)]
        int_columns = [col for col in df_pivot.columns if isinstance(col, int)]

        df_pivot = df_pivot.groupby(str_columns)[int_columns].sum().reset_index()
        selected_cols = df_pivot.loc[:, self.start_block_num : self.end_block_num]
        data = pd.concat([df_pivot[str_columns], selected_cols], axis=1)
        return data

    def _get_mean_features(self, data):
        """Generates mean features based on the part of the dataset."""
        int_columns = [col for col in data.columns if isinstance(col, int)]
        y_col_name = max(int_columns)
        x_col_names = int_columns.copy()
        x_col_names.remove(y_col_name)
        df_filtered = data.copy()
        df_filtered[x_col_names] = df_filtered[x_col_names].replace(0, np.nan)

        group_means_shop_item = (
            df_filtered.groupby(["shop_id", "item_id"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="shop_item_cnt_mean")
        )
        group_means_shop_item_category = (
            df_filtered.groupby(["shop_id", "item_category_name"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="shop_item_category_id_cnt_mean")
        )
        group_means_sity_item_id = (
            df_filtered.groupby(["shop_name", "item_id"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="sity_item_id_cnt_mean")
        )
        group_means_shop_item_category = (
            df_filtered.groupby(["shop_id", "item_category_name"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="shop_item_category_cnt_mean")
        )
        group_means_sity_category_id_cnt_mean = (
            df_filtered.groupby(["shop_name", "item_category_id"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="sity_category_id_cnt_mean")
        )
        group_means_sity_item_category = (
            df_filtered.groupby(["shop_name", "item_category_name"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="sity_item_category")
        )

        data = data.merge(group_means_shop_item, on=["shop_id", "item_id"], how="left")
        data = data.merge(
            group_means_shop_item_category,
            on=["shop_id", "item_category_name"],
            how="left",
        )
        data = data.merge(
            group_means_sity_item_id, on=["shop_name", "item_id"], how="left"
        )
        data = data.merge(
            group_means_shop_item_category,
            on=["shop_id", "item_category_name"],
            how="left",
        )
        data = data.merge(
            group_means_sity_category_id_cnt_mean,
            on=["shop_name", "item_category_id"],
            how="left",
        )
        data = data.merge(
            group_means_sity_item_category,
            on=["shop_name", "item_category_name"],
            how="left",
        )
        return data

    def _get_other_features(self, data):
        """Generates other features based on the part of the dataset."""

        data = data.fillna(0.0)
        int_columns = [col for col in data.columns if isinstance(col, int)]
        y_col_name = max(int_columns)
        x_col_names = int_columns.copy()
        x_col_names.remove(y_col_name)

        category_means = (
            data.groupby("item_category_id")["Price 1"].mean().reset_index()
        )
        bins = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 23000]
        category_means["cat_price_cat"] = pd.cut(
            category_means["Price 1"], bins=bins, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        data = pd.merge(
            data,
            category_means.drop(labels=["Price 1"], axis=1),
            on=["item_category_id"],
            how="left",
        )

        group_means_shop_item_cat_cnt_cat = (
            data.groupby(["shop_id", "cat_price_cat"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="shop_item_cat_cnt_cat")
        )
        data = data.merge(
            group_means_shop_item_cat_cnt_cat,
            on=["shop_id", "cat_price_cat"],
            how="left",
        )
        group_means_sity_item_cat_cnt_cat = (
            data.groupby(["shop_name", "cat_price_cat"])[x_col_names]
            .mean()
            .mean(axis=1)
            .reset_index(name="sity_item_cat_cnt_cat")
        )
        data = data.merge(
            group_means_sity_item_cat_cnt_cat,
            on=["shop_name", "cat_price_cat"],
            how="left",
        )

        data["year"] = (y_col_name // 12) + 2013
        data["season"] = y_col_name % 4
        data["month"] = y_col_name % 12
        return data

    def _label_cat_features(self, data):
        """Labels categorical features based on the part of the dataset."""

        data["shop_name"] = self.label_encoder.fit_transform(data["shop_name"])
        data["item_category_name"] = self.label_encoder.fit_transform(
            data["item_category_name"]
        )
        data["cat_price_cat"] = self.label_encoder.fit_transform(data["cat_price_cat"])
        data = data.fillna(0.0)
        return data

    def get_features(self, data):
        """Combines all feature generation methods and returns the final part of the dataset with features."""

        lag_price_features = self._get_price_dynamics(data)
        lag_item_data = self._lag_item_count(data)
        str_columns = [col for col in lag_item_data.columns if isinstance(col, str)]
        full_data = lag_price_features.merge(lag_item_data, on=str_columns, how="left")
        full_data = self._get_mean_features(full_data)
        full_data = self._get_other_features(full_data)
        full_data = self._label_cat_features(full_data)
        return full_data


# label_encoder = LabelEncoder()
# clip_threshold = 20
# processor = FeatureEngineering(start_block_num=0,end_block_num=34,label_encoder = label_encoder, clip_threshold = clip_threshold)
# train_data = processor.get_features(prepared_data)
