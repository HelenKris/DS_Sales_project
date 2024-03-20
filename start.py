import sys

sys.path.append("D:/Innowise/DS_Sales_project")
import config
from src.data.make_dataset import ELT

if __name__ == "__main__":
    elt_process = ELT(
        item_cat_path=config.item_cat_path,
        items_path=config.items_path,
        shops_path=config.shops_path,
        sales_train_path=config.sales_train_path,
        test_path=config.test_path,
        prepared_data_path=config.prepared_data_path,
        replace_dict=config.replace_dict,
        max_time_cnt=config.max_time_cnt,
        max_time_price=config.max_time_price,
    )
    elt_process.transform()
