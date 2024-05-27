# model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(data):
    """
    Train a Random Forest classifier model and evaluate its accuracy.

    Args:
        data (DataFrame): Input DataFrame containing features and target variable.

    Returns:
        RandomForestClassifier: Trained classifier model.
    """
    # Separate features (X) and target variable (y)
    X = data.drop("pair", axis=1)  # Features
    y = data["pair"]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Return the trained classifier model
    return rf_classifier


def predict(rf_classifier, data):
    """
    Make predictions using the trained Random Forest classifier and display feature importance's.

    Args:
        rf_classifier (RandomForestClassifier): Trained classifier model.
        data (DataFrame): Input DataFrame for making predictions.

    Returns:
        DataFrame: DataFrame containing original columns, the primary key, and the predicted labels.
        DataFrame: DataFrame containing feature importance's.
    """
    # Ensure that the features match those used during training
    X = data.drop(["pk", "pk_2"], axis=1)
    new_predictions = rf_classifier.predict(X)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Create a DataFrame to display feature importance's
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance values
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create a DataFrame with the original columns and the new predictions
    result_df = data.copy()
    result_df['predictions'] = new_predictions
    result_df = result_df[['pk_2', 'pk', 'predictions']]  # Include only 'pair', 'pk', and 'predictions'

    # Return the modified DataFrame and feature importance's DataFrame
    return result_df

