from django.shortcuts import render
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import JsonResponse,HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.db.models import Count, F, Value,Sum
import urllib, json
from django.core import serializers
from django.contrib.auth.forms import PasswordChangeForm
from datetime import date, timedelta,datetime
from dateutil.relativedelta import relativedelta

from django.db.models import Q
from django.forms.models import model_to_dict
import time
import random
import math

import decimal
from django.core.files import File
import os

import base64
import io
from django.core.files.uploadedfile import InMemoryUploadedFile

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils.timezone import localtime 
from django.core.mail import send_mail
from django.core.mail import EmailMessage


from django.template.loader import render_to_string
from django.utils.html import strip_tags

from django.views.decorators.csrf import csrf_exempt
import io
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path

#modal libs
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

import matplotlib.pyplot as plt
from bets.models import *



# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


##############
#### get all teams
def get_allteams():
    teams=[]
    # Load the CSV files
    nfl_teams_path =BASE_DIR /'data/nfl_teams.csv'   
    nfl_teams_df = pd.read_csv(nfl_teams_path)
    nfl_teams_list = nfl_teams_df.to_dict(orient='records')
    # Print the list of dictionaries
    #print(nfl_teams_list)
    teams=nfl_teams_list
    
    return teams

def get_statistics():
    stats=[]
    # Load the CSV files
    nfl_scores_path =BASE_DIR /'data/spreadspoke_scores.csv'   
    nfl_stats_df = pd.read_csv(nfl_scores_path)
    nfl_scors_list = nfl_stats_df.to_dict(orient='records')
    # Print the list of dictionaries
    #print(nfl_scors_list)
    stats=nfl_scors_list
    return stats



#clean data
def replace_nan_with_null(data):
    if isinstance(data, dict):
        return {k: replace_nan_with_null(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_null(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    return data



# get teams.
@csrf_exempt
def get_teams_data(request):
    # Assuming get_allteams() returns a list of teams
    all_teams = get_allteams()    # Replace NaN with null in the data
    # Clean the data
    cleaned_teams = replace_nan_with_null(all_teams)
   
    # Deduplicate teams by team_id
    unique_teams = {}
    for team in cleaned_teams:
        team_id = team.get("team_id")
        if team_id and team_id not in unique_teams:
            unique_teams[team_id] = team

    # Convert the dictionary back to a list
    deduplicated_teams = list(unique_teams.values()) 

    #print(deduplicated_teams)  

    #preselected list
    pre_list=['ARI', 'ATL', 'BAL' ,'BUF', 'CAR', 'CHI' ,'CIN', 'CLE', 'DAL', 'DEN' ,'DET' ,'GB','HOU' ,'IND', 'JAX', 'KC' ,'LAC', 'LAR' ,'LVR' ,'MIA', 'MIN' ,'NE' ,'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA' ,'SF' ,'TB' ,'TEN' ,'WAS']

    #new list
    newlist=[]
    for item in deduplicated_teams:
        team_id = item.get("team_id")
        if str(team_id) in pre_list:
            newlist.append(item)

    # Return the cleaned data as a JSON response
    return JsonResponse(newlist, safe=False)


# Function to parse dates and handle invalid formats
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except ValueError:
        return None  # Return None for invalid dates


# get teams.
@csrf_exempt
def get_current_statistics(request):
    # Assuming get_allteams() returns a list of teams
    all_scores = get_statistics()    # Replace NaN with null in the data
    # Clean the data
    cleaned_scores = replace_nan_with_null(all_scores)
 
    # Convert the dictionary back to a list
    deduplicated_teams = list(cleaned_scores) 
    #print(deduplicated_teams)  
    sorted_data = sorted([item for item in deduplicated_teams if parse_date(item["schedule_date"]) is not None],key=lambda x: parse_date(x["schedule_date"]),reverse=True)  # Sort in descending order

    # Return the cleaned data as a JSON response
    return JsonResponse(sorted_data, safe=False)


#test model on teams
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

    return result




#make prediction here
def create_prediction(home_team,away_team):
    # Load the CSV files
    nfl_teams_path = BASE_DIR /'data/nfl_teams.csv'
    spreadspoke_scores_path = BASE_DIR /'data/spreadspoke_scores.csv'

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
    result=test_model_on_teams(home_team, away_team,label_encoders,merged_df,scaler,xgb_model)
    print("My results: "+str(result))

    #get winner name
    winner=''
    loser=''

    if 'Away' in result:
        winner=away_team
        loser=home_team

    elif 'Home' in result:
        winner=home_team
        loser=away_team

    result_1=winner+" Team Wins"

    data={
        'result':result_1,
        'winner':winner,
        'loser':loser
    }

    return data



#get graph data
def getgraph_data(team_one,team_two):
    # Load the CSV file
    file_path = BASE_DIR /'data/spreadspoke_scores.csv'
    df = pd.read_csv(file_path)

    # Display the first few rows to understand the structure
    df.head()

    # Extracting necessary columns
    df = df[['schedule_season', 'team_home', 'score_home', 'team_away', 'score_away']]

    # Filtering for Miami Dolphins and Philadelphia Eagles matches
    teams = [team_one, team_two]
    df_filtered = df[(df['team_home'].isin(teams)) | (df['team_away'].isin(teams))]

    # Creating a dictionary to store wins per year
    win_counts = {team: {} for team in teams}

    # Counting wins per season
    for _, row in df_filtered.iterrows():
        season = int(row['schedule_season'])
        home_team = row['team_home']
        away_team = row['team_away']
        home_score = row['score_home']
        away_score = row['score_away']

        if home_team in teams:
            if season not in win_counts[home_team]:
                win_counts[home_team][season] = 0
            if home_score > away_score:
                win_counts[home_team][season] += 1

        if away_team in teams:
            if season not in win_counts[away_team]:
                win_counts[away_team][season] = 0
            if away_score > home_score:
                win_counts[away_team][season] += 1

    # Converting the dictionary to a DataFrame
    df_wins = pd.DataFrame(win_counts).fillna(0).astype(int).sort_index()

    # Resampling data for every 10 years
    df_wins_10yr = df_wins.groupby((df_wins.index // 5) * 5).sum()

    # Display the data to the user before plotting

    # Plotting the line graph
    # plt.figure(figsize=(10, 5))
    # plt.plot(df_wins_10yr.index, df_wins_10yr[team_one], marker='o', label=team_one)
    # plt.plot(df_wins_10yr.index, df_wins_10yr[team_two], marker='s', label=team_two)

    print(df_wins_10yr.index, df_wins_10yr[team_one])
    print(df_wins_10yr.index, df_wins_10yr[team_two])

    # Adding labels and title
    # plt.xlabel("Decade")
    # plt.ylabel("Total Wins")
    # plt.title("Total Wins per Decade (1966-Present)")
    # plt.legend()
    # plt.grid(True)

    # Display the plot
    #plt.show()

    # Extract separate lists
    decades = df_wins_10yr.index.tolist()
    team_one_wins = df_wins_10yr[team_one].tolist()
    team_two_wins = df_wins_10yr[team_two].tolist()
    return decades, team_one_wins, team_two_wins  # Returning separate lists



#make prediction
@csrf_exempt
def make_prediction(request):
    if request.method=="POST":
        home_team=request.POST.get('home_team')
        away_team=request.POST.get('away_team')

        home_team_name=request.POST.get('home_team_name')
        away_team_name=request.POST.get('away_team_name')
        
        client_id=request.POST.get('client_id')

        #get results
        results=create_prediction(home_team,away_team)['result']
        winner=create_prediction(home_team,away_team)['winner']
        loser=create_prediction(home_team,away_team)['loser']

        #get graph data
        #getgraph_data(home_team_name,away_team_name)
        decades, team_one_wins, team_two_wins = getgraph_data(home_team_name,away_team_name)

        graph={
            'decades':decades,
            'team_one_wins':team_one_wins,
            'team_two_wins':team_two_wins,
        }

        #create history
        cli_obj=Client.objects.filter(pk=int(client_id))[0]
        mydate=time.strftime("%d/%m/%Y")

        data_={
            'client':cli_obj,
            'home_team':home_team_name,
            'away_team':away_team_name,
            'winner':winner,
            'date':mydate,
        }
        h_obj=Predictions(**data_)
        h_obj.save()
        
        print("saved history")

    
        results={
            "message":results,
            'winner':winner,
            'loser':loser,
            'graph':graph
        }

        return JsonResponse(results,content_type='application/json',safe=False, status=200)
    
    return JsonResponse({"message":"unsuccessfull"}, status=400)



from django.contrib.auth.hashers import make_password
from django.contrib.auth.hashers import check_password

#make prediction
@csrf_exempt
def user_signup(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        message=''

        user_data={
            'username':username,
            'password':make_password(password)
        }
        #check if user exists
        if Client.objects.filter(username=username):
            message+="User Account already exists"
        else:
            #create user
            user_obj=Client(**user_data)
            user_obj.save()
            message+="Account created successfully. Please login"

        results={
            "message":message,
        }

        return JsonResponse(results,content_type='application/json',safe=False, status=200)
    
    return JsonResponse({"message":"unsuccessfull"}, status=400)



#make prediction
@csrf_exempt
def user_login(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        
        message=''
        user_data={}

        #check if user exists
        if Client.objects.filter(username=username):
            user_obj=Client.objects.filter(username=username)[0]
            if check_password(password,user_obj.password):
                message+="success"
                #get data
                user_data['username']=user_obj.username
                user_data['pk']=user_obj.pk
            else:
                message+="UserPassword is wrong."
        else:
            #create user
            message+="Account does not exist."

        results={
            "message":message,
            'user_data':user_data
        }

        return JsonResponse(results,content_type='application/json',safe=False, status=200)
    
    return JsonResponse({"message":"unsuccessfull"}, status=400)


# get teams.
@csrf_exempt
def get_predictions(request):
    if request.method=="POST":
        user_id=request.POST.get('user_id')
        report=[]
        #get user predictions
        if Client.objects.filter(pk=int(user_id)):
            cli_obj=Client.objects.filter(pk=int(user_id))[0]
            if Predictions.objects.filter(client=cli_obj):
                all_preds=Predictions.objects.filter(client=cli_obj)
                for pr in all_preds:
                    data={
                        'id':pr.pk,
                        'date':pr.date,
                        'home_team':pr.home_team,
                        'away_team':pr.away_team,
                        'winner':pr.winner,
                        'loser':pr.loser,
                    }
                    report.append(data)
        return JsonResponse(report,content_type='application/json',safe=False, status=200)
    return JsonResponse({"message":"unsuccessfull"}, status=400)
