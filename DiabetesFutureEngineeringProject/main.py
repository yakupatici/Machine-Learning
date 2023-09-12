# Problem: Developing a machine learning model that can predict whether people have diabetes or not when their characteristics are specified
# is requested. You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

# The dataset is part of a larger dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the US.
# Pima Indian women aged 21 and over living in Phoenix, the 5th largest city in the State of Arizona in the USA
#Data used for diabetes research on #. It consists of 768 observations and 8 numerical independent variables.
# The target variable is specified as "outcome"; 1 indicates a positive diabetes test result, and 0 indicates a negative diabetes test result.

# Pregnancies: Number of pregnancies
# Glucose: Glucose
#BloodPressure: Blood pressure (Diastolic)
# SkinThickness: Skin Thickness
# Insulin: Insulin.
# BMI: Body mass index.
# DiabetesPedigreeFunction: A function that calculates our probability of having diabetes based on people in our ancestry.
# Age: Age (years)
# Outcome: Information about whether the person has diabetes or not. Having the disease (1) or not (0)


# TASK 1: EXPLORATORY DATA ANALYSIS
            # Step 1: Examine the general picture.
            # Step 2: Capture numerical and categorical variables.
            # Step 3: Analyze numerical and categorical variables.
            # Step 4: Perform target variable analysis. (Average of target variable according to categorical variables, average of numerical variables according to target variable)
            # Step 5: Perform an outlier observation analysis.
            # Step 6: Perform missing observation analysis.
            # Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
            # Step 1: Take the necessary action for missing and outlier values. There are no missing observations in the data set, but Glucose, Insulin etc.
            # Observation units containing 0 values in variables may represent missing values. For example; a person's glucose or insulin level
            # cannot be 0. Considering this situation, we assign the zero values as NaN in the relevant values and then add the missing values.
            # You can apply operations.
            # Step 2: Create new variables.
            # Step 3: Perform the encoding operations.
            # Step 4: Standardize numerical variables.
            # Step 5: Create the model.






#Required libraries and functions


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


df = pd.read_csv("diabetes.csv")
df.head()
df.dtypes
df.isnull().sum()

####################
# TASK 1: EXPLORATORY DATA ANALYSIS
####################

####################
# GENERAL PICTURE
####################

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

df.head()
df.info()

####################
# CAPTURE OF NUMERICAL AND CATEGORICAL VARIABLES
####################
def grab_col_names(dataframe, cat_th=10, car_th=20):
 """

     It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
     Note: Categorical variables with numerical appearance are also included in categorical variables.

     parameters
     ------
         dataframe: dataframe
                 The dataframe from which variable names are to be retrieved
         cat_th: int, optional
                 class threshold for numeric but categorical variables
         car_th: int, optional
                 class threshold for categorical but cardinal variables

     returns
     ------
         cat_cols: list
                 Categorical variable list
         num_cols: list
                 Numeric variable list
         cat_but_car: list
                 Categorical view cardinal variable list

     examples
     ------
         import seaborn as sns
         df = sns.load_dataset("iris")
         print(grab_col_names(df))


     Notes
     ------
         cat_cols + num_cols + cat_but_car = total number of variables
         num_but_cat is inside cat_cols.

     """

 cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
 num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique ( ) < cat_th and dataframe[col].dtypes != "O"]
 cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique ( ) > car_th and dataframe[col].dtypes == "O"]
 cat_cols = cat_cols + num_but_cat
 cat_cols = [col for col in cat_cols if col not in cat_but_car]
 # num_cols
 num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
 num_cols = [col for col in num_cols if col not in num_but_cat]

 print (f"Observations: {dataframe.shape[0]}")
 print (f"Variables: {dataframe.shape[1]}")
 print (f'cat_cols: {len (cat_cols)}')
 print (f'num_cols: {len (num_cols)}')
 print (f'cat_but_car: {len (cat_but_car)}')
 print (f'num_but_cat: {len (num_but_cat)}')

 return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
cat_but_car

####################
# ANALYSIS OF CATEGORICAL VARIABLES
####################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome")

####################
# ANALYSIS OF NUMERICAL VARIABLES
####################


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

####################
# ANALYSIS OF NUMERICAL VARIABLES BY TARGET
####################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

####################
# CORRELATION
####################

# Correlation, in probability theory and statistics, indicates the direction and strength of the linear relationship between two random variables

df.corr()

#Corellation Matris
# Heat Map
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="YlGnBu") # we can use instead of "magma" , : "cubehelix"
# cmap = "RdYlBu"  , cmap = "YlGnBu"
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)
sns.heatmap()

####################
# BASE MODEL SETUP
####################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77 (TP+TN)/(TP+TN+FP+FN)
# Recall: 0.706 # how successfully the positive class was predicted TP/(TP+FN)
# Precision: 0.59 # Success of predicted values as positive class TP/(TP+FP)
# F1: 0.64 2 * ( Precision*Recall ) / ( Precision+Recall )
# Auc: 0.75

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

####################
# TASK 2: FEATURE ENGINEERING
####################

####################
# MISSING VALUE ANALYSIS
####################

# It is known that variable values other than Pregnancies and Outcome cannot be 0 in a human.
# Therefore, an action decision must be made regarding these values. Values that are 0 can be assigned NaN.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# We went to each of the variables with 0 in the observation units and replaced the observation values containing 0 with NaN.

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

#Missing Data Analysis

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)