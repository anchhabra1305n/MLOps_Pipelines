#!/usr/bin/env python
# coding: utf-8

# 

# # 1. Import Libraraies

# In[1]:


# this will help in making the Python code more structured automatically (good coding practice)
#%load_ext nb_black
#mlflow.set_experiment(experiment_id="0")
#mlflow.autolog()
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)


# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


# # 2.Data Generation

# In[2]:


import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic data with specific features
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
data["churn"] = y

# Save data for future use
data.to_csv("churn_data.csv", index=False)


# # 3. EDA

# In[3]:


import pandas as pd

# Load data
churn = pd.read_csv("churn_data.csv")


# In[4]:


# copying data to another variable to avoid any changes to original data
data = churn.copy()


# In[5]:


data.head(5)


# In[6]:


data.tail(5)


# In[7]:


print('Number of records of the Churn data is ',data.shape[0])
print('Number of feature of the Churn data is ',data.shape[1])


# In[8]:


data.info()


# In[9]:


data.describe().T


# In[10]:


data.isnull().sum()


# In[11]:


# filtering object type columns
con_columns = data.describe(include=["float64"]).columns
con_columns


# ## 3.1 Univariate Analysis

# In[12]:


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    #print('Histogram and Bobplot view............. for the feature of ',feature)
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[13]:


for i in con_columns:
    #print('Histogram and Bobplot view............. for the feature of ',i)
    histogram_boxplot(data, i, bins=100)


# In[14]:


missing_count_df = data.isnull().sum() # the count of missing value
value_count_df = data.isnull().count() # the count of all values
missing_percentage_df = round(missing_count_df/value_count_df*100,2) # the percentage of missing values
missing_df = pd.DataFrame({'count' : missing_count_df, 'percentage' : missing_percentage_df }) # create a dataframe
barchart = missing_df.plot.bar(y='percentage',rot=90,figsize=(30,10))
for index, percentage in enumerate(missing_percentage_df):
    barchart.text(index,percentage,str(percentage) + '%')


# In[15]:


plt.figure(figsize=(20,10))
data.boxplot()
plt.show()


#  ## 3.2 Outlier treatment

# In[16]:


df_num = data.select_dtypes(include = ['float64', 'int64'])
lstnumericcolumns = list(df_num.columns.values)
len(lstnumericcolumns)


# In[17]:


### Outlier treatment :
def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[18]:


for column in data[lstnumericcolumns].columns:
    lr,ur=remove_outlier(data[column])
    data[column]=np.where(data[column]>ur,ur,data[column])
    data[column]=np.where(data[column]<lr,lr,data[column])


# ## 3.3 after outlier treatmnt Box plot and hist plot

# In[19]:


plt.figure(figsize=(20,10))
data.boxplot()
plt.show()


# In[20]:


data_plot=data[data.dtypes[data.dtypes!='object'].index]
fig=plt.figure(figsize=(12,7))
for i in range(0,len(data_plot.columns)):
   ax=fig.add_subplot(5,3,i+1)
   sns.distplot(data_plot[data_plot.columns[i]])
   ax.set_title(data_plot.columns[i],color='Blue')
plt.tight_layout()


# ## 3.3 Bivariate Analysis

# In[21]:


sns.pairplot(data, hue='churn')


# In[22]:


plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# In[23]:


### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


# In[24]:


for i in con_columns:
    #print('Histogram and Bobplot view............. for the feature of ',i)
    distribution_plot_wrt_target(data, i, "churn")


# # Creating training and test sets

# In[25]:


X = data.drop(["churn"], axis=1)
Y = data["churn"]

X = pd.get_dummies(X, drop_first=True)

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)


# In[26]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# In[27]:


import pandas as pd

# Load data
data = pd.read_csv("churn_data.csv")

# Preprocessing steps (e.g., handle missing values, encode categorical features)
# data["missing_value"] = data["feature_2"].fillna(-1)
# data = pd.get_dummies(data, columns=["feature_7"])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(data.drop("churn", axis=1), data["churn"], test_size=0.2, random_state=42)


# In[28]:


import mlflow.sklearn

#if __name__ == "__main__":
    
X = data.drop(["churn"], axis=1)
Y = data["churn"]
X = pd.get_dummies(X, drop_first=True)
# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
lr =RandomForestClassifier(random_state=42)
mlflow.set_experiment(experiment_id="0")
mlflow.autolog()
lr.fit(X, y)

from sklearn.metrics import accuracy_score

# Evaluate model performance
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

score = lr.score(X, y)
print("Score: %s" % score)
mlflow.log_metric("score", score)
mlflow.sklearn.log_model(lr, "model")

from sklearn.model_selection import GridSearchCV

# Define parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [50,75,100, 200, 300],
    "max_depth": [5, 10, 15]
}

# Tune model using GridSearchCV
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Get best model and score
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f"Best Model Score: {best_score}")

mlflow.log_metric("best_score", best_score)
mlflow.sklearn.log_model(best_model, "model")

print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# # Load model

# In[33]:


X.to_csv("input_new.csv",index=False)
input=pd.read_csv("input_new.csv")


# In[29]:


import mlflow.pyfunc

model =mlflow.pyfunc.load_model("runs:/d94ed5e555614b0faea846186ed68abe/model")


# In[41]:


predictions=model.predict(input)
my_predictions=pd.DataFrame(predictions)
my_predictions.columns =['churn']

output=input.join(my_predictions)
output.to_csv("output.csv", index=False)


# In[42]:


output.head()


# In[ ]:




