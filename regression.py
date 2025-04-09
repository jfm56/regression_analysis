import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_enhanced_visualizations(df, feature_columns, target_column, model_coef):
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Encode categorical variables for correlation
    categorical_columns = plot_df[feature_columns].select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        plot_df[col] = le.fit_transform(plot_df[col].astype(str))
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = plot_df[feature_columns + [target_column]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced Scatter Plots with Trend Lines
    numerical_features = [
        'Pregnancy Test', 'Birth Control', 'Folic Acid',
        'Prenatal Vitamins', 'Cigarettes', 'Wine'
    ]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.regplot(data=plot_df, x=feature, y=target_column, 
                   scatter_kws={'alpha':0.5}, 
                   line_kws={'color': 'red'})
        plt.title(f'{feature} vs Pregnancy')
        plt.xlabel(feature)
        plt.ylabel('Pregnant (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance Plot
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': np.abs(model_coef[0])
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, y='Feature', x='Importance', 
                palette='viridis')
    plt.title('Absolute Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEnhanced visualizations saved as:")
    print("1. correlation_heatmap.png - Shows correlations between all features")
    print("2. scatter_plots.png - Scatter plots with trend lines for top features")
    print("3. feature_importance.png - Bar plot of feature importance")

def perform_regression(excel_path, target_column, feature_columns=None):
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # If feature columns not specified, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Handle categorical variables
    X = df[feature_columns].copy()
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Apply label encoding to categorical columns
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Prepare target variable
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model (using Logistic Regression since this is a classification problem)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print("\nClassification Results:")
    print("----------------------")
    print(f"Accuracy Score: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nFeature Importance:")
    for feature, coef in zip(feature_columns, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    
    # Create enhanced visualizations
    create_enhanced_visualizations(df, feature_columns, target_column, model.coef_)

if __name__ == "__main__":
    # File path and target column
    excel_path = "target.xlsx"
    target_column = "PREGNANT"
    
    # Specify feature columns (excluding unnamed and target columns)
    feature_columns = [
        'Implied Gender',
        'Home/Apt/ PO Box',
        'Pregnancy Test',
        'Birth Control',
        'Feminine Hygiene',
        'Folic Acid',
        'Prenatal Vitamins',
        'Prenatal Yoga',
        'Body Pillow',
        'Ginger Ale',
        'Sea Bands',
        'Stopped buying ciggies',
        'Cigarettes',
        'Smoking Cessation',
        'Stopped buying wine',
        'Wine',
        'Maternity Clothes'
    ]
    
    perform_regression(excel_path, target_column, feature_columns)
