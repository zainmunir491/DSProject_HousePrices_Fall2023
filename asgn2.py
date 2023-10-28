## Hakim Shahzad Tiwana : 20L-1119
## Muhammad Zain : 20L-2168
## Noor Fatima : 20L-0990

from numpy import dtype
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import folium.plugins # The Folium Javascript Map Library

data = pd.read_csv("./zameen-property-data.csv",
                   header=0,
                   usecols=["city", "latitude","longitude","property_type","price","area","purpose","bedrooms","date_added"],
                   )
data

# Data Wrangling and Cleaning

#Obtain dataset for proprties in lahore thar are for sale
lahoreDataSet=data.query('city=="Lahore"')
lahoreDataSet=lahoreDataSet.query('purpose=="For Sale"')
lahoreDataSet

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


# (DATA TRANSFORMATION)
lahoreDataSet=convert_area_to_marla_unit(lahoreDataSet)
lahoreDataSet

#Get age for each property in seconds (DATA TRANSFORMATION)
AgeList=[]
for date in lahoreDataSet["date_added"]:
  AgeList.append(datetime.now().timestamp()-datetime.strptime(date, '%m-%d-%Y').timestamp())
lahoreDataSet["Age"] = AgeList
lahoreDataSet["Age"]

#Extract only house properties
lahoreDataSet = lahoreDataSet.query("property_type == 'House'")
#Drop non numeric columns (DIMENSIONALITY REDUCTION)
lahoreDataSet = lahoreDataSet.drop(["city","property_type","area","date_added","purpose"],axis=1)
lahoreDataSet

"""
Excludes any outliers which are based on the parameters passed
"""
def exclude_outliers(dframe,price,area,bedrooms):

  filtered=dframe.query('price>1000000 and price <'+str(price))
  filtered=filtered.query('bedrooms>0 and bedrooms<'+str(bedrooms))
  filtered=filtered.query("area_in_marlas>0 and area_in_marlas<"+str(area))
  return filtered


#(DATA CLEANING)
lahoreDataSet=exclude_outliers(lahoreDataSet,150000000,81,8)
lahoreDataSet

#(STANDARDIZATION) since we have different units of measurements.
scaler = StandardScaler()
scaler.fit(lahoreDataSet)
lahoreDataSet_standardized = pd.DataFrame(scaler.transform(lahoreDataSet), columns=list(lahoreDataSet.columns))
lahoreDataSet_standardized

#Check to see if there is any missing values (DATA VALIDATION)
print(lahoreDataSet_standardized.isna().sum())

# (DATA VALIDATION)
lahoreDataSet_standardized.info()

#Get dataframe numerical statistics (DATA VALIDATION)
lahoreDataSet_standardized.describe()

#Converting to CSV
lahoreDataSet_standardized.to_csv('clean_standardized_LahoreDataSet.csv')
lahoreDataSet.to_csv('clean_LahoreDataSet.csv')

# Exploratory Data Analysis

## Summary Statistics

lahoreDataSet.describe()

lahoreDataSet.corr()

#Making a subset of the DF to get every 15th row so that the visualization is more understandable
SubLahoreDataSet = lahoreDataSet.iloc[::15, :]

## Univariate

### Price

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

## Bivariant

### Price

for columns in SubLahoreDataSet.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(SubLahoreDataSet[columns], SubLahoreDataSet['price'])
    plt.title('Scatter Plot of Price vs '+ columns)
    plt.xlabel(columns)
    plt.ylabel('Price')
    plt.show()

### Correlation


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

### Maps


long=lahoreDataSet.longitude.mean()
lat=lahoreDataSet.latitude.mean()
SF_COORDINATES = (lat, long)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = lahoreDataSet[['latitude', 'longitude']].astype('float').dropna().to_numpy()
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
sf_map.add_child(heatmap)

cluster = folium.plugins.MarkerCluster()
for _, r in lahoreDataSet[['latitude', 'longitude', 'price']].tail(5000).dropna().iterrows():
    cluster.add_child(
        folium.Marker([float(r["latitude"]), float(r["longitude"])], popup=r['price']))

sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
sf_map.add_child(cluster)
sf_map

