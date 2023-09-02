
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# %%
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# %%
df_train.head()

# %%
df_test.head()

# %%
print('Train Data',df_train.shape)
print('Test Data',df_test.shape)

# %% [markdown]
# * The train data consists of 8,523 training examples with 12 features.
# * The test data consists of 5,681 training examples with 11 features.

# %% [markdown]
# * **Finding some basic information about the features of the data.**

# %%
df_train.info()

# %%
df_train.boxplot()

# %%
print('Number of trainings examples:', len(df_train),'\n')
df_train.describe().T.style.background_gradient(cmap='Blues')

# %%
# List of numerical features:
numerical = df_train.select_dtypes(include = ['int64', 'Int64','float64']).dtypes.index
numerical

# %%
# List of categorical features
cat_features = df_train.select_dtypes(include = ['object']).dtypes.index
cat_features

# %%
def UVA_Categorical(data, cat):
  plt.figure(figsize = (10,6))
  sns.countplot(cat, data = data)
  plt.xlabel(cat,fontsize = 14, fontweight = 'bold')
  plt.ylabel('Count',fontsize = 14, fontweight = 'bold')
  plt.title('Value counts: \n{}'.format(df_train[cat].value_counts(normalize = True)))

  # Rotating xticklabels
  if len(data[cat].value_counts()) > 7:
    X = plt.gca().xaxis
    for item in X.get_ticklabels():
      item.set_rotation(90)
  plt.show()

# %%
UVA_Categorical(df_train,'Item_Fat_Content')

# %%
total_low_fat = 0.597090 + 0.037076 + 0.013141
total_low_fat

# %%
UVA_Categorical(df_train, 'Item_Type')

# %%
UVA_Categorical(df_train, 'Outlet_Identifier')

# %%
UVA_Categorical(df_train,'Outlet_Size')

# %%
UVA_Categorical(df_train, 'Outlet_Location_Type')

# %%
UVA_Categorical(df_train, 'Outlet_Type')

# %%
df_train['source'] = 'train'
df_test['source'] = 'test'
df=pd.concat([df_train,df_test], ignore_index=True)

# %%
df.info()

# %%
df.isnull().sum()

# %%
plt.figure(figsize = (10,6))
sns.heatmap(df.isnull(), yticklabels=False,cbar = False,cmap ='viridis')

# %%
# Percentage of missing values:

def missing_percent():
  miss_item_weight = (df['Item_Weight'].isnull().sum()/len(df))*100
  miss_Outlet_Size = (df['Outlet_Size'].isnull().sum()/len(df))*100

  print('% of missing values in Item_Weight: ' + str(miss_item_weight))
  print('% of missing values in Outlet_Size: ' +str(miss_Outlet_Size))
    
missing_percent()

# %%
df['Item_Weight'].fillna(df['Item_Weight'].mean(),inplace=True) 

# %%

df['Outlet_Size'].value_counts()

# %%
df['Outlet_Size'].fillna('Medium', inplace=True)

# %%
df.isnull().sum()

# %%
df.Item_Visibility.value_counts

# %%
df['Item_Visibility'].replace(0.0,value=np.nan,inplace=True)  # first replace 0 with nan values

# %%
# fill nan values with corresponding item identifier mean value
df['Item_Visibility']=df['Item_Visibility'].fillna(df.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))

# %%
plt.figure(figsize=(7,5))
sns.countplot('Item_Fat_Content',data=df)

# %%
df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace=True)

# %%
plt.figure(figsize=(7,5))
sns.countplot('Item_Fat_Content',data=df)

# %%
# Store data for future prediction
test_pred = df.loc[df['source'] == 'test']
test_pred.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
test_pred.head()

# %%
from sklearn.preprocessing import LabelEncoder
categ = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
le = LabelEncoder()
df[categ] = df[categ].apply(le.fit_transform)

# %%
df.head()

# %%
df.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)

# %%
df.head()

# %%
train = df.loc[df['source'] == 'train']
test = df.loc[df['source'] == 'test']

# %%
train.drop(['source'],axis=1,inplace=True)
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)

# %%
train.head()

# %%
test.head()

# %%
x = train.drop(columns="Item_Outlet_Sales")
y = train["Item_Outlet_Sales"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state = 0)


# %%
# create empty set to srore accuracies of all modes and later use for comparison
model_comparison = {} 

# %%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

# %%
y_pred = lr.predict(X_test)

# %%
y_pred

# %%
model_comparison['Linear Regression'] = [lr.score(X_train,y_train)*100,r2_score(y_test,y_pred)*100]

print("Linear Regression\n\nAccuracy: {}%".format(round(lr.score(X_train,y_train)*100)))
print("r2 score: {}%".format(round(r2_score(y_test,y_pred)*100)))

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

# %%
y_pred

# %%
model_comparison['Decision Tree'] = [tree.score(X_train,y_train)*100,r2_score(y_test,y_pred)*100]

print("Decision Tree\n\nAccuracy: {}%".format(round(tree.score(X_train,y_train)*100)))
print("r2 score: {}%".format(round(r2_score(y_test,y_pred)*100)))

# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=400,max_depth=6,min_samples_leaf=100,n_jobs=4)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

# %%
y_pred

# %%
model_comparison['Random Forest'] = [rf.score(X_train,y_train)*100,r2_score(y_test,y_pred)*100]

print("Random Forest\n\nAccuracy: {}%".format(round(rf.score(X_train,y_train)*100)))
print("r2 score: {}%".format(round(r2_score(y_test,y_pred)*100)))

# %%
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators = 100, learning_rate=0.05)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

# %%
y_pred

# %%
model_comparison['XGBoost Regressor'] = [xgb.score(X_train,y_train)*100,r2_score(y_test,y_pred)*100]

print("XGBoost Regressor\n\nAccuracy: {}%".format(round(xgb.score(X_train,y_train)*100)))
print("r2 score: {}%".format(round(r2_score(y_test,y_pred)*100)))

# %%
model_comparison_df = pd.DataFrame.from_dict(model_comparison).T
model_comparison_df.columns = ['Accuracy', "r2_score"]
model_comparison_df = model_comparison_df.sort_values('Accuracy', ascending=True)
model_comparison_df.style.background_gradient(cmap='Blues')

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
fig = go.Figure(data=[
    go.Bar(name='r2_score', y=model_comparison_df.index, x=model_comparison_df['r2_score'], orientation='h', marker_color='#7baaf7'),
    go.Bar(name='Accuracy', y=model_comparison_df.index, x=model_comparison_df['Accuracy'], orientation='h', marker_color='#4285f4')
])
fig.update_layout(barmode='group')
fig.show()

# %% [markdown]
# # Test Data 

# %%
test.head()


# %%
pred = xgb.predict(test)

# %%
pred

# %%
test_pred.head()

# %%
test_pred["Predicted_Item_Outlet_Sale"] = pred

# %%
test_pred

# %%
test_pred.to_csv("submission.csv",index=False)

# %%


# %%
# sample input values for prediction

# 20.750000	0	0.007565	13	107.8622	1999	1	0	1
# 8.300000	1	0.038428	4	87.3198	2007	1	1	1

# %%
ls=[]
for i in test.columns:
    s = float(input(f"Enter the {i}:"))
    ls.append(s)

# %%
xgb.predict(np.array(ls).reshape(1,-1))

# %%
ls1=[]
for i in test.columns:
    s = float(input(f"Enter the {i}:"))
    ls1.append(s)

# %%
xgb.predict(np.array(ls1).reshape(1,-1))

# %%
import joblib
import pickle

joblib.dump(xgb,"BigMart_model.sav")

# %%
pickle.dump(xgb,open('BigMart_model.pkl','wb'))

# %%









