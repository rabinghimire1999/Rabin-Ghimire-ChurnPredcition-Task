# Customer Churn Prediction

This project involves developing a machine learning model to predict customer churn using a given dataset. The task includes data preprocessing, exploratory data analysis (EDA), model development, hyperparameter tuning, and evaluation. The dataset contains various features related to customer behavior and demographics, with the target variable indicating whether a customer has churned (yes/no).

## Getting Started

To set up and run this project, follow the steps below:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/rabinghimire1999/Rabin-Ghimire-ChurnPrediction-Task.git
    cd Rabin-Ghimire-ChurnPredcition-Task
    ```

2. **Activate a virtual environment and install dependencies**:

    Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

    Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the respective Python scripts**:

    - **Data Preprocessing**:

        ```bash
        python data_preprocessing.py
        ```

    - **Exploratory Data Analysis (EDA)**:

        ```bash
        python eda.py
        ```

    - **Model Development**:

        ```bash
        python model_development.py
        ```

    - **Model Evaluation**:

        ```bash
        python model_evaluation.py
        ```

## Project Structure

- `data_preprocessing.py`: Script for loading the data, handling missing values, converting categorical variables to numerical format, and scaling the features.
- `eda.py`: Script for performing exploratory data analysis, including visualizations of the target variable distribution, correlation heatmap, and pair plots.
- `model_development.py`: Script for training different machine learning models (Logistic Regression, Random Forest, Gradient Boosting), including hyperparameter tuning using GridSearchCV.
- `model_evaluation.py`: Script for evaluating the trained models using metrics such as accuracy, precision, recall, F1 score, ROC AUC score, and confusion matrix.

## Requirements

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

All required libraries can be installed via `pip install -r requirements.txt`.

## Author

- Rabin Ghimire