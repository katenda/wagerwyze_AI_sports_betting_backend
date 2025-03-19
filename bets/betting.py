
import pandas as pd
import seaborn as sns
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import io
from io import BytesIO
from IPython.display import clear_output
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
# encoding
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from random import randrange

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train an improved model using XGBoost
from xgboost import XGBClassifier
from tabulate import tabulate
from pathlib import Path

############
####  test model

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


def test_model_on_teams(home_team, away_team,label_encoders,merged_df,scaler,xgb_model):
    # Ensure team names are valid
    if home_team not in label_encoders["team_home_id"].classes_ or away_team not in label_encoders["team_away_id"].classes_:
        print("Invalid team name. Please check the team names from the dataset.")
        return
    
    # Encode team names
    home_team_id = label_encoders["team_home_id"].transform([home_team])[0]
    away_team_id = label_encoders["team_away_id"].transform([away_team])[0]

    # Get the latest known stats for these teams
    latest_home_stats = merged_df[merged_df["team_home_id"] == home_team_id].iloc[-1:]
    latest_away_stats = merged_df[merged_df["team_away_id"] == away_team_id].iloc[-1:]

    # Handle missing stats (in case of limited history)
    home_last5_score_home = latest_home_stats["team_home_id_last5_score_home"].values[0] if not latest_home_stats.empty else 0
    home_last5_score_away = latest_home_stats["team_home_id_last5_score_away"].values[0] if not latest_home_stats.empty else 0
    away_last5_score_home = latest_away_stats["team_away_id_last5_score_home"].values[0] if not latest_away_stats.empty else 0
    away_last5_score_away = latest_away_stats["team_away_id_last5_score_away"].values[0] if not latest_away_stats.empty else 0

    # Define input features (with default spread, weather, etc.)
    input_features = np.array([[home_team_id, away_team_id, -3.5, 45.0,  # Spread & Over/Under Line
                                65, 5, 50, 0,  # Weather & Stadium Neutral (0 for home advantage)
                                home_last5_score_home, home_last5_score_away, 
                                away_last5_score_home, away_last5_score_away]])

    # Standardize features
    input_features = scaler.transform(input_features)

    # Predict game outcome
    prediction = xgb_model.predict(input_features)[0]
    
    # Interpret the result
    result = "Home Team Wins" if prediction == 1 else "Away Team Wins"

    # Prepare table data
    table_data = [
        ["Home Team", home_team],
        ["Away Team", away_team],
        ["Predicted Outcome", result]
    ]

    # Print tabular result
    print(tabulate(table_data, headers=["Category", "Value"], tablefmt="grid"))



#############
##main function
###############

def main():
    print("BETTING  ANALYTICS")
    
    # Load the CSV files
    nfl_teams_path = "/Users/enockkatenda4/Projects/betting_app/inputs/nfl_teams.csv"
    spreadspoke_scores_path = "/Users/enockkatenda4/Projects/betting_app/inputs/spreadspoke_scores.csv"

    nfl_teams_df = pd.read_csv(nfl_teams_path)
    spreadspoke_scores_df = pd.read_csv(spreadspoke_scores_path)

    # Display basic info and first few rows of each dataset
    nfl_teams_df.info(), nfl_teams_df.head(), spreadspoke_scores_df.info(), spreadspoke_scores_df.head()


    # Convert schedule_date to datetime format
    spreadspoke_scores_df["schedule_date"] = pd.to_datetime(spreadspoke_scores_df["schedule_date"])

    # Fill missing values in numeric columns with their median values
    numeric_cols = ["spread_favorite", "weather_temperature", "weather_wind_mph", "weather_humidity"]
    spreadspoke_scores_df[numeric_cols] = spreadspoke_scores_df[numeric_cols].fillna(spreadspoke_scores_df[numeric_cols].median())

    # Fill missing categorical columns with a placeholder
    spreadspoke_scores_df["team_favorite_id"].fillna("Unknown", inplace=True)
    spreadspoke_scores_df["over_under_line"].fillna("45.0", inplace=True)  # Using 45 as a reasonable median assumption

    # Convert over_under_line to numeric
    spreadspoke_scores_df["over_under_line"] = pd.to_numeric(spreadspoke_scores_df["over_under_line"], errors="coerce")

    # Merge with nfl_teams_df to map team identifiers
    merged_df = spreadspoke_scores_df.merge(
        nfl_teams_df[["team_name", "team_id"]],
        left_on="team_home",
        right_on="team_name",
        how="left"
    ).rename(columns={"team_id": "team_home_id"}).drop(columns=["team_name"])

    merged_df = merged_df.merge(
        nfl_teams_df[["team_name", "team_id"]],
        left_on="team_away",
        right_on="team_name",
        how="left"
    ).rename(columns={"team_id": "team_away_id"}).drop(columns=["team_name"])

    # Check the cleaned and merged data
    print("Preprocessed Data\n", merged_df)


    # Create point differential feature
    merged_df["point_differential"] = merged_df["score_home"] - merged_df["score_away"]

    # Encode categorical variables
    label_encoders = {}
    for col in ["team_home_id", "team_away_id", "team_favorite_id", "stadium"]:
        le = LabelEncoder()
        merged_df[col] = le.fit_transform(merged_df[col].astype(str))
        label_encoders[col] = le

    # Select features and target
    features = [
        "team_home_id", "team_away_id", "spread_favorite", "over_under_line",
        "weather_temperature", "weather_wind_mph", "weather_humidity", "stadium_neutral"
    ]
    X = merged_df[features]
    y = (merged_df["point_differential"] > 0).astype(int)  # 1 if home team wins, 0 otherwise

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    accuracy, report

    # Check for missing values in features
    missing_values = X.isnull().sum()
    missing_values

    # Fill missing values in over_under_line with the median
    X["over_under_line"].fillna(X["over_under_line"].median(), inplace=True)

    # Recheck missing values
    missing_values_after = X.isnull().sum()
    missing_values_after

    # Re-split dataset after handling missing values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model again
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    accuracy, report

    # Feature Engineering: Adding team performance metrics

    # Sort data by date to maintain chronological order
    merged_df = merged_df.sort_values(by="schedule_date")

    # Create rolling averages for last 5 games for home and away teams
    for team_col in ["team_home_id", "team_away_id"]:
        for stat in ["score_home", "score_away", "point_differential"]:
            merged_df[f"{team_col}_last5_{stat}"] = merged_df.groupby(team_col)[stat].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    # Fill any remaining NaN values that resulted from rolling calculations
    merged_df.fillna(0, inplace=True)

    # Select updated features
    features += [
        "team_home_id_last5_score_home", "team_home_id_last5_score_away", "team_home_id_last5_point_differential",
        "team_away_id_last5_score_home", "team_away_id_last5_score_away", "team_away_id_last5_point_differential",
    ]

    # Prepare the dataset
    X = merged_df[features]
    y = (merged_df["point_differential"] > 0).astype(int)  # 1 if home team wins, 0 otherwise

    # Re-split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_pred = xgb_model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    accuracy, report


    # Convert schedule_date to datetime format
    spreadspoke_scores_df["schedule_date"] = pd.to_datetime(spreadspoke_scores_df["schedule_date"])

    # Fill missing values in numeric columns with their median values
    numeric_cols = ["spread_favorite", "weather_temperature", "weather_wind_mph", "weather_humidity"]
    spreadspoke_scores_df[numeric_cols] = spreadspoke_scores_df[numeric_cols].fillna(spreadspoke_scores_df[numeric_cols].median())

    # Fill missing categorical columns with a placeholder
    spreadspoke_scores_df["team_favorite_id"].fillna("Unknown", inplace=True)
    spreadspoke_scores_df["over_under_line"].fillna("45.0", inplace=True)  # Using 45 as a reasonable median assumption
    spreadspoke_scores_df["over_under_line"] = pd.to_numeric(spreadspoke_scores_df["over_under_line"], errors="coerce")

    # Merge with nfl_teams_df to map team identifiers
    merged_df = spreadspoke_scores_df.merge(
        nfl_teams_df[["team_name", "team_id"]],
        left_on="team_home",
        right_on="team_name",
        how="left"
    ).rename(columns={"team_id": "team_home_id"}).drop(columns=["team_name"])

    merged_df = merged_df.merge(
        nfl_teams_df[["team_name", "team_id"]],
        left_on="team_away",
        right_on="team_name",
        how="left"
    ).rename(columns={"team_id": "team_away_id"}).drop(columns=["team_name"])

    # Feature Engineering: Adding team performance metrics
    merged_df = merged_df.sort_values(by="schedule_date")

    # Create rolling averages for last 5 games for home and away teams
    for team_col in ["team_home_id", "team_away_id"]:
        for stat in ["score_home", "score_away"]:
            merged_df[f"{team_col}_last5_{stat}"] = merged_df.groupby(team_col)[stat].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    # Fill any remaining NaN values
    merged_df.fillna(0, inplace=True)

    # Create point differential feature
    merged_df["point_differential"] = merged_df["score_home"] - merged_df["score_away"]

    # Encode categorical variables
    label_encoders = {}
    for col in ["team_home_id", "team_away_id", "team_favorite_id", "stadium"]:
        le = LabelEncoder()
        merged_df[col] = le.fit_transform(merged_df[col].astype(str))
        label_encoders[col] = le

    # Select updated features
    features = [
        "team_home_id", "team_away_id", "spread_favorite", "over_under_line",
        "weather_temperature", "weather_wind_mph", "weather_humidity", "stadium_neutral",
        "team_home_id_last5_score_home", "team_home_id_last5_score_away",
        "team_away_id_last5_score_home", "team_away_id_last5_score_away"
    ]

    # Prepare the dataset
    X = merged_df[features]
    y = (merged_df["point_differential"] > 0).astype(int)  # 1 if home team wins, 0 otherwise

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an improved model using XGBoost
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_pred = xgb_model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    accuracy, report
    print(label_encoders["team_home_id"].classes_)
    

    # **Example Usage**
    test_model_on_teams("LVR", "WAS",label_encoders,merged_df,scaler,xgb_model)


if __name__=="__main__":
    main()  



