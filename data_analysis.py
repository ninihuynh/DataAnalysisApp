import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File at {file_path} not found.")
    
def clean_data(df):
    """
    Cleans the data by handling missing values, duplicates, and data types.
    """
    # Handling missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Removing duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def explore_data(df):
    """
    Basic exploratory data analysis.
    """
    print("Basic Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nCorrelation Matrix:")
    print(df.select_dtypes(include='number').corr())
    
    return df

def visualize_data(df):
    """
    Visualizes the data with different plots.
    """
    numeric_df = df.select_dtypes(include='number')

    sns.pairplot(numeric_df)
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def build_model(df, target_column):
    """
    Builds and evaluates a simple linear regression model.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return model

def generate_report(df, model, target_column):
    """
    Generates a summary report for the analysis.
    """
    report = {
        'Data Summary': df.describe(),
        'Correlation Matrix': df.corr(),
        'Model Coefficients': model.coef_,
        'Intercept': model.intercept_
    }
    print("\nAnalysis Report:")
    for section, content in report.items():
        print(f"\n{section}:")
        print(content)

def main():
    print("Welcome to the Data Analysis App!")
    file_path = input("Enter the CSV file path: ")
    
    # Load and clean data
    try:
        df = load_data(file_path)
        df = clean_data(df)
        
        # Explore the data
        explore_data(df)
        
        # Visualize the data
        visualize_data(df)
        
        # Build and evaluate a simple model
        target_column = input("Enter the target column for prediction (e.g., 'Price'): ")

        # Encode categorical target if it's non-numeric
        from sklearn.preprocessing import LabelEncoder

        if df[target_column].dtype == 'object':
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])

        model = build_model(df, target_column)
        
        # Generate the analysis report
        generate_report(df, model, target_column)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()
