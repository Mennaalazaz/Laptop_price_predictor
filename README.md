# Laptop Price Predictor

This project aims to predict the price of laptops based on their various features. The process involves data cleaning, exploratory data analysis (EDA), feature engineering, model building, and evaluation using a dataset of laptop specifications and prices.

## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Dataset](#dataset)
3.  [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Model Building](#model-building)
6.  [Results](#results)
7.  [How to Use the Model](#how-to-use-the-model)
8.  [Dependencies](#dependencies)
9.  [File Structure](#file-structure)
10. [Future Work](#future-work)

## Project Goal
The main objective of this project is to develop a regression model that can accurately predict the price of a laptop given its specifications.

## Dataset
The project uses a dataset named `laptop_data.csv`.
The initial dataset contains the following columns:
*   `Unnamed: 0`: An index column (later dropped).
*   `Company`: Laptop manufacturer.
*   `TypeName`: Type of laptop (e.g., Notebook, Ultrabook, Gaming).
*   `Inches`: Screen size in inches.
*   `ScreenResolution`: Screen resolution and panel type.
*   `Cpu`: CPU model and speed.
*   `Ram`: RAM in GB.
*   `Memory`: Storage type and capacity.
*   `Gpu`: Graphics Processing Unit.
*   `OpSys`: Operating System.
*   `Weight`: Weight in kg.
*   `Price`: Price of the laptop (target variable).

## Data Preprocessing and Feature Engineering
Several preprocessing and feature engineering steps were performed:
1.  **Initial Cleaning:**
    *   Dropped the `Unnamed: 0` column.
    *   Cleaned the `Ram` column by removing 'GB' and converting to an integer.
    *   Cleaned the `Weight` column by removing 'kg' and converting to a float.
2.  **Screen Features:**
    *   Extracted `Touchscreen` (binary: 1 if Touchscreen, 0 otherwise) from `ScreenResolution`.
    *   Extracted `IPS` (binary: 1 if IPS panel, 0 otherwise) from `ScreenResolution`.
    *   Split `ScreenResolution` into `X_Res` (horizontal resolution) and `Y_Res` (vertical resolution).
    *   Calculated `ppi` (pixels per inch) using `X_Res`, `Y_Res`, and `Inches`.
    *   Dropped `ScreenResolution`, `X_Res`, `Y_Res`, and `Inches` columns after creating `ppi`.
3.  **CPU Features:**
    *   Extracted `CPU Name` (e.g., "Intel Core i5") from the `Cpu` column.
    *   Categorized `CPU Name` into `cpu Brand` ('Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor').
    *   Dropped the original `Cpu` and `CPU Name` columns.
4.  **Memory Features:**
    *   Cleaned and parsed the `Memory` column to extract capacities for different storage types.
    *   Created separate columns for `HDD` and `SSD` capacities (in GB).
    *   Dropped columns related to 'Hybrid' and 'Flash_Storage' to simplify the model, keeping only HDD and SSD.
    *   Dropped the original `Memory` column.
5.  **GPU Features:**
    *   Extracted `Gpu_Brand` (first word of the GPU description, e.g., "Intel", "Nvidia", "AMD") from the `Gpu` column.
    *   Dropped the original `Gpu` column.
6.  **Operating System Features:**
    *   Categorized `OpSys` into a simpler `OS` column ('Windows', 'Mac', 'Others/No OS/Linux').
    *   Dropped the original `OpSys` column.
7.  **Target Variable Transformation:**
    *   The `Price` column (target variable) was found to be right-skewed. A log transformation (`np.log(df['Price'])`) was applied to make its distribution more normal, which often helps in improving model performance.

## Exploratory Data Analysis (EDA)
*   Visualized the distribution of `Price`, `Inches`, `Weight`, and the log-transformed `Price`.
*   Examined relationships between categorical features (`Company`, `TypeName`, `Touchscreen`, `IPS`, `cpu Brand`, `Gpu_Brand`, `OS`, `Ram`) and `Price` using bar plots.
*   Examined relationships between numerical features (`Inches`, `Weight`) and `Price` using scatter plots.
*   Calculated correlations between numerical features and `Price`. `Ram`, `SSD`, and `ppi` showed significant positive correlations.

## Model Building
1.  **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets.
2.  **Preprocessing Pipeline:**
    *   A `ColumnTransformer` was used to apply `OneHotEncoder` (with `drop='first'` to avoid multicollinearity) to the following categorical features: `Company`, `TypeName`, `cpu Brand`, `Gpu_Brand`, `OS`.
    *   Numerical features (`Ram`, `Weight`, `Touchscreen`, `IPS`, `ppi`, `HDD`, `SSD`) were passed through without transformation in this step.
3.  **Models Trained:** Several regression models were trained and evaluated using a `Pipeline` that included the preprocessing step:
    *   Linear Regression
    *   Ridge Regression (alpha=10)
    *   Lasso Regression (alpha=0.001)
    *   KNeighbors Regressor (n_neighbors=3)
    *   Decision Tree Regressor (max_depth=8)
4.  **Evaluation Metrics:** Models were evaluated using R2 Score and Mean Absolute Error (MAE) on the log-transformed price.

## Results
The Decision Tree Regressor (with `max_depth=8`) was the last model trained and the one pickled for future use. Its performance on the test set was:
*   **R2 Score:** Approximately 0.80
*   **Mean Absolute Error (MAE):** Approximately 0.20 (on log-transformed price)

This indicates that the model can explain about 80% of the variance in the log-transformed laptop prices. The MAE of 0.20 on the log scale means that, on average, the model's predictions (in log terms) are off by about 0.20.

## How to Use the Model
The trained pipeline (including the preprocessor and the Decision Tree model) and the processed DataFrame have been saved using `pickle`.
*   `pipe.pkl`: The trained scikit-learn pipeline.
*   `df.pkl`: The processed DataFrame (useful for understanding feature names and structure expected by the model).

To make predictions on new data:
1.  Load the pipeline:
    ```python
    import pickle
    import numpy as np
    import pandas as pd # For creating a DataFrame for new data

    pipe = pickle.load(open('pipe.pkl', 'rb'))
    # df_processed = pickle.load(open('df.pkl', 'rb')) # To see column order/names
    ```
2.  Prepare your new data. It should be a Pandas DataFrame with the same columns (and in the same order) as the `X_train` used to fit the pipeline. These columns are:
    `['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi', 'cpu Brand', 'HDD', 'SSD', 'Gpu_Brand', 'OS']`
    Ensure data types match (e.g., 'Ram' as int, 'Weight' as float, 'ppi' as float, 'HDD'/'SSD' as int).

    ```python
    # Example new data point
    new_laptop_data = pd.DataFrame({
        'Company': ['Apple'],
        'TypeName': ['Ultrabook'],
        'Ram': [8], # int
        'Weight': [1.37], # float
        'Touchscreen': [0], # int (0 or 1)
        'IPS': [1], # int (0 or 1)
        'ppi': [226.983005], # float
        'cpu Brand': ['Intel Core i5'],
        'HDD': [0], # int
        'SSD': [128], # int
        'Gpu_Brand': ['Intel'],
        'OS': ['Mac']
    })
    ```
3.  Make predictions:
    ```python
    log_price_prediction = pipe.predict(new_laptop_data)
    price_prediction = np.exp(log_price_prediction) # Convert back from log scale

    print(f"Predicted Log Price: {log_price_prediction[0]}")
    print(f"Predicted Actual Price: {price_prediction[0]}")
    ```

## Dependencies
*   Python 3.x
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   pickle (standard library)

You can typically install these using pip:
`pip install pandas numpy matplotlib seaborn scikit-learn`

## File Structure


.
├── laptop_price_predictor.ipynb # Jupyter notebook with the analysis and model training
├── laptop_data.csv # Original dataset
├── df.pkl # Pickled processed DataFrame
├── pipe.pkl # Pickled trained model pipeline
└── README.md # This file

## Future Work
*   **Hyperparameter Tuning:** Optimize the parameters of the chosen model (or other models) using techniques like GridSearchCV or RandomizedSearchCV.
*   **Try More Advanced Models:** Explore models like Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), or Neural Networks.
*   **Advanced Feature Engineering:**
    *   Extract CPU clock speed or generation.
    *   More detailed GPU categorization.
*   **Cross-Validation:** Implement k-fold cross-validation for more robust model evaluation.
*   **Deployment:** Create a simple web application (e.g., using Flask or Streamlit) to serve the model for interactive predictions.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
