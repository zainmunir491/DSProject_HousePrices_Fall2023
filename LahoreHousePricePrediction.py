#!/usr/bin/env python
# coding: utf-8

# ## Hakim Shahzad Tiwana : 20L-1119
# ## Muhammad Zain : 20L-2168
# ## Noor Fatima : 20L-0990
# 

# In[245]:


from numpy import dtype
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

 #!pip install gradio==2.6.4
import gradio as gr


# In[246]:


data = pd.read_csv("./zameen-property-data.csv",
                   header=0,
                   usecols=["city", "latitude","longitude","property_type","price","area","purpose","bedrooms","date_added"],
                   )
data


# # Data Wrangling and Cleaning

# In[247]:


#Obtain dataset for proprties in lahore thar are for sale
lahoreDataSet=data.query('city=="Lahore"')
lahoreDataSet=lahoreDataSet.query('purpose=="For Sale"')
lahoreDataSet


# In[248]:


'''
Converts string area in the Dataframe to numeric values in unit of marlas
'''
def convert_area_to_marla_unit(dataframe):
  area = dataframe['area']
  numeric_area=[]
  for x in area:
    num,_,unit=x.partition(" ")
    if(unit=="Kanal"):
      num=num.replace(",","")
      newNum=float(num)*20
    else:
      newNum=float(num)
    numeric_area.append(newNum)
  dataframe["area_in_marlas"]=numeric_area
  return dataframe


# In[249]:


# (DATA TRANSFORMATION)
lahoreDataSet=convert_area_to_marla_unit(lahoreDataSet)
lahoreDataSet


# In[250]:


#Get age for each property in seconds (DATA TRANSFORMATION)
AgeList=[]
for date in lahoreDataSet["date_added"]:
  AgeList.append(datetime.now().timestamp()-datetime.strptime(date, '%m-%d-%Y').timestamp())
lahoreDataSet["Age"] = AgeList
lahoreDataSet["Age"]


# In[251]:


#Extract only house properties
lahoreDataSet = lahoreDataSet.query("property_type == 'House'")
#Drop non numeric columns (DIMENSIONALITY REDUCTION)
lahoreDataSet = lahoreDataSet.drop(["city","property_type","area","date_added","purpose"],axis=1)
lahoreDataSet


# In[252]:


"""
Excludes any outliers which are based on the parameters passed
"""
def exclude_outliers(dframe,price,area,bedrooms):

  filtered=dframe.query('price>1000000 and price <'+str(price))
  filtered=filtered.query('bedrooms>0 and bedrooms<'+str(bedrooms))
  filtered=filtered.query("area_in_marlas>0 and area_in_marlas<"+str(area))
  return filtered


# In[253]:


#(DATA CLEANING)
lahoreDataSet=exclude_outliers(lahoreDataSet,150000000,81,8)
lahoreDataSet


# In[254]:


#(STANDARDIZATION) since we have different units of measurements.
scaler = StandardScaler()
scaler.fit(lahoreDataSet)
lahoreDataSet_standardized = pd.DataFrame(scaler.transform(lahoreDataSet), columns=list(lahoreDataSet.columns))
lahoreDataSet_standardized


# In[255]:


#Check to see if there is any missing values (DATA VALIDATION)
print(lahoreDataSet_standardized.isna().sum())

# (DATA VALIDATION)
lahoreDataSet_standardized.info()


# In[256]:


#Get dataframe numerical statistics (DATA VALIDATION)
lahoreDataSet_standardized.describe()


# In[257]:


#Converting to CSV
lahoreDataSet_standardized.to_csv('clean_standardized_LahoreDataSet.csv')
lahoreDataSet.to_csv('clean_LahoreDataSet.csv')


# # Exploratory Data Analysis

# ## Summary Statistics

# In[258]:


lahoreDataSet.describe()


# In[259]:


lahoreDataSet.corr()


# In[260]:


#Making a subset of the DF to get every 15th row so that the visualization is more understandable
SubLahoreDataSet = lahoreDataSet.iloc[::15, :]


# ## Univariate

# ### Price

# In[261]:


plt.figure(figsize=(10, 6))
hist, bins, _=plt.hist(SubLahoreDataSet.price, bins=50, edgecolor='black')
plt.title('Histogram for Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
for i in range(len(hist)):
    if hist[i] > 0:
        plt.text(bins[i], hist[i]+5, f'{int(hist[i])}', ha='left', va='bottom',rotation='vertical')


plt.figure(figsize=(10, 6))
plt.boxplot(SubLahoreDataSet.price)
plt.title('Box Plot for price')
plt.ylabel('Values')
plt.show()



# In[262]:


#Age in Seconds
plt.figure(figsize=(10, 6))
Age_years = SubLahoreDataSet.Age/(60*60*24*365.25)
hist, bins, _ = plt.hist(Age_years, bins=50, edgecolor='black')
plt.title('Histogram for Age (in Years)')
plt.xlabel('Age')
plt.ylabel('Frequency')
for i in range(len(hist)):
    if hist[i] > 0:
        plt.text(bins[i], hist[i]+5, f'{int(hist[i])}', ha='left', va='bottom',rotation='vertical')

plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(Age_years)
plt.title('Box Plot for Age')
plt.ylabel('Values')
plt.show()


# In[263]:


#Area
plt.figure(figsize=(10, 6))
hist, bins, _= plt.hist(SubLahoreDataSet["area_in_marlas"], bins=50, edgecolor='black')
plt.title('Histogram for Area (in marlas)')
plt.xlabel('Area')
plt.ylabel('Frequency')
for i in range(len(hist)):
    if hist[i] > 0:
        plt.text(bins[i], hist[i]+5, f'{int(hist[i])}', ha='left', va='bottom',rotation='vertical')

plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(SubLahoreDataSet["area_in_marlas"])
plt.title('Box Plot for Area (in marlas)')
plt.ylabel('Values')
plt.show()


Piedf = SubLahoreDataSet
threshold=50
Piedf['area_in_marlas'] = Piedf['area_in_marlas'].round()
area_counts = Piedf['area_in_marlas'].value_counts()
other_mask = area_counts < threshold
area_counts['Other'] = area_counts[other_mask].sum()
area_counts = area_counts[area_counts>threshold]
area_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(8, 8))
plt.title('Distribution of Area in Marlas')
plt.show()


# In[264]:


#Number of bedrooms
bedroom_counts = SubLahoreDataSet['bedrooms'].value_counts()
bars=plt.bar(bedroom_counts.index, bedroom_counts)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.title('Distribution of Properties by Number of Bedrooms')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

plt.show()


Piedf = SubLahoreDataSet
threshold=50
Piedf['bedrooms'] = Piedf['bedrooms'].round()
room_counts = Piedf['bedrooms'].value_counts()
other_mask = room_counts < threshold
room_counts['Other'] = room_counts[other_mask].sum()
room_counts = room_counts[room_counts>threshold]
room_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(8, 8))
plt.title('Distribution of Bedrooms')
plt.show()


# ## Bivariant

# ### Price

# In[265]:


for columns in SubLahoreDataSet.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(SubLahoreDataSet[columns], SubLahoreDataSet['price'])
    plt.title('Scatter Plot of Price vs '+ columns)
    plt.xlabel(columns)
    plt.ylabel('Price')
    plt.show()


# ### Correlation

# In[266]:


import seaborn as sns
target_variable = 'price'

# Selecting other variables (excluding the target)
features = SubLahoreDataSet.columns[SubLahoreDataSet.columns != target_variable]

# Correlation Matrix
correlation_matrix = SubLahoreDataSet.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title(f'Correlation Heatmap with'+target_variable)
plt.show()


# ### Maps
# 

# In[267]:


import folium
import folium.plugins # The Folium Javascript Map Library
long=lahoreDataSet.longitude.mean()
lat=lahoreDataSet.latitude.mean()
SF_COORDINATES = (lat, long)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = lahoreDataSet[['latitude', 'longitude']].astype('float').dropna().to_numpy()
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
sf_map.add_child(heatmap)


# In[268]:


cluster = folium.plugins.MarkerCluster()
for _, r in lahoreDataSet[['latitude', 'longitude', 'price']].tail(5000).dropna().iterrows():
    cluster.add_child(
        folium.Marker([float(r["latitude"]), float(r["longitude"])], popup=r['price']))

sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
sf_map.add_child(cluster)
sf_map


# # Model Training

# In[ ]:





# In[269]:


#Divide data set into training and testing sets
X = lahoreDataSet_standardized.drop(["price"],axis=1)
y = lahoreDataSet_standardized["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[270]:


def ModelEvaluation(Model):
    
    #Fit data into Model
    Model.fit(X_train,y_train)
    
    #Gradients and Intercept
    print("Inercept : ")
    print(Model.intercept_)
    ModelCoeff = pd.DataFrame(Model.coef_, X.columns, columns=['coefficient'])
    print("Coefficients : ")
    print(ModelCoeff)

    #Predictions
    y_predict = Model.predict(X_test)
    Predictions = pd.DataFrame({'Test':y_test, 'Prediction':y_predict}).head(10)
    print("Predictions : ")
    print(Predictions)

    #Evaluate Results
    MAE = metrics.mean_absolute_error(y_predict, y_test)
    MSE = metrics.mean_squared_error(y_predict, y_test)
    RMSE = np.sqrt(MSE)
    R2= round(metrics.r2_score(y_test, y_predict), 2)
   
    EvaluationResults = pd.DataFrame([MAE, MSE, RMSE,R2], index=['MAE', 'MSE', 'RMSE','R2'], columns=['Metrics'])
    print("Evaluation Results : ")
    print(EvaluationResults)
    print("R-squared Accuracy shows : " + str(R2*100) + "%")
    


# In[271]:


#Liner Regression
Lreg = LinearRegression()
ModelEvaluation(Lreg)


# In[272]:


#Lasso Regression
lasso = Lasso(alpha=0.01)
ModelEvaluation(lasso)


# In[273]:


#Ridge Regression
ridge = Ridge()
ModelEvaluation(ridge)


# In[274]:


#Elastic Net Regression
elasticNet = ElasticNet(alpha=0.01,l1_ratio=0.01)
ModelEvaluation(elasticNet)


# In[275]:


'''
Very poor results from stochastic gradient descent
'''
#from sklearn.linear_model import SGDRegressor
#GdRegr = SGDRegressor(loss="epsilon_insensitive",penalty="elasticnet",alpha=0.75, l1_ratio=0.75)
#ModelEvaluation(GdRegr)


# In[276]:


# Define a function to take inputs and predict the house price using Linear Regression or Lasso Model
def predict_price(latitude, longitude, bedrooms, area_in_marlas, age_years, age_months, age_days, model_type):
    # Set bounds for input values
    lat_min, lat_max = 31.252821, 33.749464
    lon_min, lon_max = 73.035393, 74.501450
    max_area = 80
    
    # Check if inputs are within bounds
    if latitude < lat_min or latitude > lat_max:
        return "Error: Latitude must be between {} and {}.".format(lat_min, lat_max)
    if longitude < lon_min or longitude > lon_max:
        return "Error: Longitude must be between {} and {}.".format(lon_min, lon_max)
    if bedrooms <= 0:
        return "Error: House must have bedrooms, Number of bedrooms cannot be zero."
    if area_in_marlas > max_area:
        return "Error: Area in marlas cannot be more than {}.".format(max_area)

    # Create input dataframe
    userinput = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "bedrooms": [bedrooms],
        "area_in_marlas": [area_in_marlas],
        "Age": [(age_years*365*24*3600) + (age_months*30*24*3600) + (age_days*24*3600)]
        
    })
    input_data = X.copy(deep=True)
    input_data = pd.concat([input_data,userinput],ignore_index=True)
    scaler_I = StandardScaler()
    scaler_I.fit(input_data)
    scaled_input_data = scaler_I.transform(input_data)
    
    # Make prediction based on model_type
    if model_type == 'Linear':
        scaled_predicted_price = Lreg.predict(scaled_input_data)[len(input_data)-1]
    elif model_type == 'Lasso':
        scaled_predicted_price = lasso.predict(scaled_input_data)[len(input_data)-1]
    elif model_type == 'Ridge':
        scaled_predicted_price = ridge.predict(scaled_input_data)[len(input_data)-1]
    elif model_type == 'ElasticNet':
        scaled_predicted_price = elasticNet.predict(scaled_input_data)[len(input_data)-1]
    else:
        return "Error: Invalid model_type. Choose from Linear, Lasso, Ridge or ElasticNet."
    scaledSet= lahoreDataSet_standardized.copy(deep=True)
    userinput["price"] = scaled_predicted_price
    pred_price = scaler.inverse_transform(userinput)[0][5]
    # Check if predicted price is negative
    if pred_price < 0:
        return "Cannot make prediction using {} Model.".format(model_type)

    # Return result
    print(pred_price)
    return str(pred_price)

#---------------------------------------------------------------------------------------------------------------------

# Use Gradio to build the interface for the application and connect it with the function
inputs = [
    gr.inputs.Number(label="Latitude (Min: 31.252821,  Max: 33.749464)", default=32),
    gr.inputs.Number(label="Longitude (Min: 73.035393,  Max: 74.501450)", default=74),
    gr.inputs.Number(label="Bedrooms (Max 8)", default=4),
    gr.inputs.Number(label="Area in marlas (Max 80)", default=20),
    gr.inputs.Number(label="Age (years)", default=6.0),
    gr.inputs.Number(label="Age (months)", default=11.0),
    gr.inputs.Number(label="Age (days)", default=5.0),
    gr.inputs.Radio(['Linear', 'Lasso','Ridge','ElasticNet'], label="Choose a model for prediction")
]

output = gr.outputs.Textbox(label="Predicted House Price (in Pakistani Rupees)")

# Create interface for both models
gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=output,
    title="Lahore House Price Predictor",
    description="Predict the price of a house in Lahore based on its location, size, and age.",
).launch(inline=False)


# In[ ]:





# In[ ]:





# In[242]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




