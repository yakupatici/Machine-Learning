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

check_df(df)

# Getting Categoric, numeric function
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols

# ADVANCED FUNCTIONAL EDA
# 1. Outliers
# 2. Missing Values
# 3. Feature Extraction
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling

#########################
# 1.Outliers
#########################
# Note that Outliers are generally more commonly
# associated with numerical variables, but in categorical variables,
# rare or unexpected values can also be considered outliers, and in such cases,
# different methods can be used to address them.

for col in num_cols:
    print(col, check_outlier(df, col))

# Outlier graph of each outlier column one by one
for col in num_cols:
    plt.title(col)
    sns.boxplot(df[col])
    plt.show(block=True)

# changing minimum and maximum limits
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.1, q3=0.9))

for col in num_cols:
    if check_outlier(df, col, q1=0.1, q3=0.9):
        replace_with_thresholds(df, col, q1=0.1, q3=0.9)

#############################################
# 2.Analysis of Categorical Variables
#############################################

for col in cat_cols:
    cat_summary(df, col, plot=True)

#############################################
# 3.Analysis of Numerical Variables
#############################################

for col in num_cols:
    num_summary(df, col, plot=True)

#############################################
# 4. Analysis of Target Variable
#############################################

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

#############################################
# 5. Analysis of Correlation
#############################################
high_correlated_cols(df, plot=True)



df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

#############################################
# 3. Feature Extraction
#############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols

new_num_cols = [col for col in num_cols if col not in ["Salary", "Years"]]

df[new_num_cols] = df[new_num_cols] + 1


df.columns = [col.upper() for col in df.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# RATIO OF VARIABLES

# CAREER RUNS RATIO
df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
# CAREER BAT RATIO
df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
# CAREER HITS RATIO
df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
# CAREER HMRUN RATIO
df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
# CAREER RBI RATIO
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
# CAREER WALKS RATIO
df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]
df["NEW_C_HIT_RATE"] = df["CHITS"] / df["CATBAT"]
# PLAYER TYPE : RUNNER
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHITS"]
# PLAYER TYPE : HIT AND RUN
df["NEW_C_HIT-AND-RUN"] = df["CRUNS"] / df["CHITS"]
# MOST VALUABLE HIT RATIO IN HITS
df["NEW_C_HMHITS_RATIO"] = df["CHMRUN"] / df["CHITS"]
# MOST VALUABLE HIT RATIO IN ALL SHOTS
df["NEW_C_HMATBAT_RATIO"] = df["CATBAT"] / df["CHMRUN"]

#Annual Averages
df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]


# PLAYER LEVEL X DIVISION

df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Junior-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Junior-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Mid-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Mid-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Senior-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Senior-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Expert-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Expert-West"

# Player Promotion to Next League
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"



num_cols
cat_cols
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoding

binary_cols = [col for col in df.columns if
               df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# Rare Encoding

rare_analyser(df,"SALARY", cat_cols)
df = rare_encoder(df, 0.01, cat_cols)


# 6. One-Hot Encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove("SALARY")

# 7. Robust-Scaler
for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])


######################################################
# Multiple Linear Regression
######################################################
X = df.drop("SALARY", axis=1)
y = df[["SALARY"]]
y
X
##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
# b + w*x
# sabit (b - bias)
reg_model.intercept_

# Coefficients (w - weights)
reg_model.coef_

#Linear regression y_hat = b + w*x
np.inner(X_train.iloc[2, :].values ,reg_model.coef_) + reg_model.intercept_
y_train.iloc[2]

np.inner(X_train.iloc[4, :].values ,reg_model.coef_) + reg_model.intercept_
y_train.iloc[4]

##########################
# Prediction
##########################

##########################
# Evaluating Prediction Accuracy
##########################


# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# smape = mean_absolute_error(y_train, y_pred) / y_train.mean()


# TRAIN RKARE
reg_model.score(X_train, y_train)



# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)


# 10-Fold Cross-Validation(CV) RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, y,
                                 cv=10,
                                 scoring="neg_mean_squared_error"))) #negative-mean-squared-error from sckitlearn




