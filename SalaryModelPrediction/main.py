###################################################
# PROJECT: SALARY PREDICTION WITH MACHINE LEARNING
###################################################

# Business Problem

# Can a machine learning project be conducted to predict the salaries of baseball players
# based on salary information and career statistics for the year 1986?

# Dataset Story

# This dataset was originally obtained from the StatLib library at Carnegie Mellon University.
# The dataset is part of the data used in the Poster Session of the ASA Graphics Section in 1988.
# Salary data was originally obtained from Sports Illustrated on April 20, 1987.
# The 1986 and career statistics were obtained from the 1987 Baseball Encyclopedia Update,
# published by Collier Books, Macmillan Publishing Company, New York.

# AtBat: Number of times at bat during the 1986-1987 season with a baseball bat
# Hits: Number of hits during the 1986-1987 season
# HmRun: Number of home runs during the 1986-1987 season
# Runs: Number of runs scored for the team during the 1986-1987 season
# RBI: Number of runs batted in by the batter when hitting
# Walks: Number of errors made by the opposing player
# Years: Number of years the player has played in the major league (years)
# CAtBat: Number of times the player has hit the ball during his career
# CHits: Number of hits made by the player during his career
# CHmRun: Number of home runs made by the player during his career
# CRuns: Number of runs scored for the team during the player's career
# CRBI: Number of players who were made to run during the player's career
# CWalks: Number of errors made by the player against the opposing player during his career
# League: A factor with levels A and N that shows the league the player played in until the end of the season
# Division: A factor with levels E and W that shows the position the player played at the end of 1986
# PutOuts: Cooperation within the game with your teammates
# Assists: Number of assists made by the player during the 1986-1987 season
# Errors: Number of errors made by the player during the 1986-1987 season
# Salary: Player's salary received during the 1986-1987 season (in thousands)
# NewLeague: A factor with levels A and N that shows the player's league at the beginning of the 1987 season

############################################
# Required Libraries and Functions
############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")


#detailed view

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


data = pd.read_csv("hitters.csv")
df = data.copy()

# TASK 1: EXPLORATORY DATA ANALYSIS

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df.info
df.columns

# ADVANCED FUNCTIONAL EDA
# 1. Outliers
# 2. Missing Values
# 3. Feature Extraction
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling