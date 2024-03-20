item_cat_path = "./data.dvc/item_categories.csv"
items_path = "./data.dvc/items.csv"
shops_path = "./data.dvc/shops.csv"
sales_train_path = "./data.dvc/sales_train.csv"
test_path = "./data.dvc/test.csv"
prepared_data_path = "./data.dvc/prepared_data.csv"

X_train_path = "./data/processed/X_train.csv"
y_train_path = "./data/processed/y_train.csv"
X_val_path = "./data/processed/X_val.csv"
y_val_path = "./data/processed/y_val.csv"
X_test_path = "./data/processed/X_test.csv"
best_grid_model_save_path = "./src/models/best_grid_model.pkl"
xgb_model_save_path = "./src/models/xgb_model.pkl"


# Dictionary of system duplicate store ids that need to be replaced
replace_dict = {0: 57, 1: 58, 11: 10, 40: 39}

# Number of month of test observation
# test_block_num = 34

# Parameter for estimating the daily outlier, if exceeded by a specified number of times,it will be deleted
max_time_price = 10
max_time_cnt = 100

# Parameter for target clipping
clip_threshold = 20
