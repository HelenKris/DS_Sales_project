Predict Future Sales
==============================

## Project Structure

```

DS Project/
├── notebooks/
│   ├── Practice 1.1.DQCipynb
│   └── Practice_1_2_EDA.ipynb
├── data/
│   ├── raw  <- The original, immutable data dump.
│   └── processed  <- The final, canonical data sets for modeling.
├── models/ <- Trained and serialized models
├── .gitignore
├── README.md
├── requirements.txt <- The requirements file for reproducing the analysis environment
└──  src/
    ├── data <- Scripts to download or generate data
    │    ├── make_dataset.py
    │    └── make_train_val_test.py
    ├── features <- Scripts to generate additional features
    │    └── build_features.py
    ├── models <- Scripts to train models and then use trained models to make predictions
    │    ├── predict_model.py
    │    ├── select_hyperparam.py
    │    └── train_model.py
    ├── utils.py
    └──__init__.py
```
