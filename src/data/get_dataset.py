import pandas as pd
import click


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=5)
@click.argument("output_path", type=click.Path())
def main(input_paths, output_path):
    item_cat_path, items_path, shops_path, sales_train_path, test_path = input_paths

    elt = ELT(
        item_cat_path, items_path, shops_path, sales_train_path, test_path, output_path
    )
    elt.transform()


class ELT:
    def __init__(
        self,
        item_cat_path,
        items_path,
        shops_path,
        sales_train_path,
        test_path,
        prepared_data_path,
    ):
        self.item_cat_path = item_cat_path
        self.items_path = items_path
        self.shops_path = shops_path
        self.sales_train_path = sales_train_path
        self.test_path = test_path
        self.prepared_data_path = prepared_data_path

    def transform(self):
        df_item_cat, df_items, df_shops, sales_train, test = self._extract_data()
        transformed_data = self._transform_data(
            df_item_cat, df_items, df_shops, sales_train, test
        )
        filtered_data = self._remove_outliers(transformed_data)
        grouped_data = self._get_grouped_data(filtered_data)
        self._load_data(grouped_data, self.prepared_data_path)

    def _extract_data(self):
        df_item_cat = pd.read_csv(self.item_cat_path)
        df_items = pd.read_csv(self.items_path)
        df_shops = pd.read_csv(self.shops_path)
        sales_train = pd.read_csv(self.sales_train_path)
        test = pd.read_csv(self.test_path)
        return df_item_cat, df_items, df_shops, sales_train, test

    def _transform_data(self, df_item_cat, df_items, df_shops, sales_train, test):
        data = sales_train.copy()
        replace_dict = {0: 57, 1: 58, 11: 10, 40: 39}
        for old_value, new_value in replace_dict.items():
            data.loc[data["shop_id"] == old_value, "shop_id"] = new_value
        test_block_num = sales_train["date_block_num"].max() + 1

        test["date_block_num"] = test_block_num
        main_features = ["date_block_num", "shop_id", "item_id"]
        data = pd.concat(
            [data, test.drop("ID", axis=1)], ignore_index=True, keys=main_features
        )
        data = data.fillna(0)

        common_column = None
        for df in [df_items, df_item_cat, df_shops]:
            if common_column is None:
                common_column = set(data.columns) & set(df.columns)
            else:
                common_column = list(set(data.columns) & set(df.columns))

            data = pd.merge(data, df, on=list(common_column), how="left")

        # Delete only one position with the maximum price and all positions with a negative price
        max_price_item = data.loc[data["item_price"].idxmax(), "item_name"]
        data = data[data["item_name"] != max_price_item]
        data = data.query("item_price >= 0")

        data = data.drop(labels=["date"], axis=1)
        data["item_category_name"] = data["item_category_name"].astype(str)
        data["shop_name"] = data["shop_name"].astype(str)
        data["item_name"] = data["item_name"].astype(str)
        return data

    def _remove_outliers(self, data):
        max_time_cnt = 10
        max_time_price = 100
        # At the level of daily observations, remove observations x times larger than the average observation of the sale of this product
        mean_quantity_by_item = data.groupby("item_name")["item_cnt_day"].mean()
        mean_price_by_item = data.groupby("item_name")["item_price"].mean()
        outliers_quantity = data[
            data["item_cnt_day"]
            > max_time_cnt * data["item_name"].map(mean_quantity_by_item)
        ]
        outliers_price = data[
            data["item_price"]
            > max_time_price * data["item_name"].map(mean_price_by_item)
        ]
        data = data.drop(outliers_quantity.index)
        data = data.drop(outliers_price.index)
        return data

    def _get_grouped_data(self, data):
        data = data.groupby(
            [
                "date_block_num",
                "shop_id",
                "item_category_id",
                "item_id",
                "item_category_name",
                "shop_name",
            ],
            as_index=False,
        ).agg({"item_price": ["mean"], "item_cnt_day": ["sum"]})
        data.columns = [
            "date_block_num",
            "shop_id",
            "item_category_id",
            "item_id",
            "item_category_name",
            "shop_name",
            "item_price",
            "item_cnt",
        ]
        return data

    def _load_data(self, data, prepared_data_path):
        data.to_csv(prepared_data_path, index=False)


if __name__ == "__main__":
    main()
