#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statistics as st
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('train-2.csv')


# In[3]:


df


# In[4]:


df.columns


# In[5]:


df['Airline'].value_counts()


# In[6]:


df['Departure_City'].value_counts()


# In[7]:


df['Arrival_City'].value_counts()


# In[8]:


df['Distance'].value_counts()


# In[9]:


df['Departure_Time'].value_counts()


# In[10]:


df['Arrival_Time'].value_counts()


# In[11]:


df['Duration'].value_counts()


# In[12]:


df['Aircraft_Type'].value_counts()


# In[13]:


df['Number_of_Stops'].value_counts()


# In[14]:


df['Day_of_Week'].value_counts()


# In[15]:


df['Month_of_Travel'].value_counts()


# In[16]:


df['Holiday_Season'].value_counts()


# In[17]:


df['Demand'].value_counts()


# In[18]:


df['Weather_Conditions'].value_counts()


# In[19]:


df['Passenger_Count'].value_counts()


# In[20]:


df['Promotion_Type'].value_counts()


# In[21]:


df['Fuel_Price'].value_counts()


# In[22]:


df.isnull().sum()


# In[23]:


# Calculate the mode of the "Airline" column
mode_value = df['Airline'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Airline'].fillna(mode_value, inplace=True)


# In[24]:


# Calculate the mode of the "Airline" column
mode_value = df['Departure_City'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Departure_City'].fillna(mode_value, inplace=True)


# In[25]:


# Calculate the mode of the "Airline" column
mode_value = df['Arrival_City'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Arrival_City'].fillna(mode_value, inplace=True)


# In[26]:


# Calculate the mean of the "Distance" column
mean_value = df['Distance'].mean()

# Fill missing values in the "Distance" column with the mean
df['Distance'].fillna(mean_value, inplace=True)


# In[27]:


# Calculate the mode of the "Airline" column
mode_value = df['Aircraft_Type'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Aircraft_Type'].fillna(mode_value, inplace=True)


# In[28]:


# Calculate the mode of the "Airline" column
mode_value = df['Day_of_Week'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Day_of_Week'].fillna(mode_value, inplace=True)


# In[29]:


# Calculate the mode of the "Airline" column
mode_value = df['Month_of_Travel'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Month_of_Travel'].fillna(mode_value, inplace=True)


# In[30]:


# Calculate the mode of the "Airline" column
mode_value = df['Holiday_Season'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Holiday_Season'].fillna(mode_value, inplace=True)


# In[31]:


# Calculate the mode of the "Airline" column
mode_value = df['Demand'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Demand'].fillna(mode_value, inplace=True)


# In[32]:


# Calculate the mode of the "Airline" column
mode_value = df['Weather_Conditions'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Weather_Conditions'].fillna(mode_value, inplace=True)


# In[33]:


# Calculate the mode of the "Airline" column
mode_value = df['Promotion_Type'].mode()[0]

# Fill missing values in the "Airline" column with the mode
df['Promotion_Type'].fillna(mode_value, inplace=True)


# In[34]:


# Calculate the mean of the "Distance" column
mean_value = df['Fuel_Price'].mean()

# Fill missing values in the "Distance" column with the mean
df['Fuel_Price'].fillna(mean_value, inplace=True)


# In[35]:


df.isnull().sum()


# In[36]:


df.head(10)


# In[37]:


df.dtypes


# In[ ]:





# In[38]:


plt.boxplot(df['Distance'])


# In[ ]:





# In[ ]:





# In[39]:


plt.boxplot(df['Passenger_Count'])


# In[40]:


plt.boxplot(df['Fuel_Price'])


# In[41]:


plt.boxplot(df['Flight_Price'])


# In[42]:


plt.boxplot(df['Number_of_Stops'])


# In[43]:


plt.boxplot(df['Duration'])


# # OUTLIERS

# In[44]:


q1 = df['Flight_Price'].quantile(0.25)
q3 = df['Flight_Price'].quantile(0.75)
iqr = q3 - q1
upper_threshold = q3 + 1.5 * iqr
lower_threshold = q1 - 1.5 * iqr
upper_threshold, lower_threshold


# In[45]:


df['Flight_Price'] = df['Flight_Price'].clip(upper_threshold, lower_threshold)


# In[46]:


plt.boxplot(df['Flight_Price'])


# In[47]:


# q1 = df['Number_of_Stops'].quantile(0.25)
# q3 = df['Number_of_Stops'].quantile(0.75)
# iqr = q3 - q1
# upper_threshold = q3 + 1.5 * iqr
# lower_threshold = q1 - 1.5 * iqr
# upper_threshold, lower_threshold


# In[48]:


# df['Number_of_Stops'] = df['Number_of_Stops'].clip(upper_threshold, lower_threshold)


# In[49]:


# plt.boxplot(df['Number_of_Stops'])


# In[50]:


df


# In[51]:


df.drop_duplicates()


# # EDA

# In[52]:


plt.hist(df['Distance'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Distance')
plt.show()


# In[53]:


plt.hist(df['Flight_Price'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Flight_Price')
plt.ylabel('Frequency')
plt.title('Histogram of Flight_Price')
plt.show()


# In[54]:


plt.hist(df['Fuel_Price'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Fuel_Price')
plt.ylabel('Frequency')
plt.title('Histogram of Fuel_Price ')
plt.show()


# In[55]:


plt.hist(df['Passenger_Count'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Passenger_Count')
plt.ylabel('Frequency')
plt.title('Histogram of Passenger_Count ')
plt.show()


# In[56]:


plt.hist(df['Number_of_Stops'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Number_of_Stops')
plt.ylabel('Frequency')
plt.title('Histogram of Number_of_Stops ')
plt.show()


# In[57]:


plt.hist(df['Departure_Time'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Departure_Time')
plt.ylabel('Frequency')
plt.title('Histogram of Departure_Time ')
plt.show()


# In[58]:


plt.hist(df['Arrival_Time'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Arrival_Time')
plt.ylabel('Frequency')
plt.title('Histogram of Arrival_Time  ')
plt.show()


# In[59]:


plt.hist(df['Duration'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Duration  ')
plt.show()


# In[60]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Promotion_Type'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Promotion_Type')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[61]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Airline'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Airline')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[62]:


# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming you have already loaded your DataFrame df

# # Count the occurrences of each unique category
# category_counts = df['Departure_City'].value_counts()

# # Extract the category names and their respective counts
# categories = category_counts.index
# counts = category_counts.values

# # Create a bar plot
# plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
# plt.bar(categories, counts)

# # Optionally, you can set labels for the x-axis and y-axis
# plt.xlabel('Departure_City')
# plt.ylabel('Count')

# # Optionally, you can rotate the x-axis labels for better readability
# plt.xticks(rotation=45)

# # Show the plot
# plt.show()


# In[63]:


# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming you have already loaded your DataFrame df

# # Count the occurrences of each unique category
# category_counts = df['Arrival_City'].value_counts()

# # Extract the category names and their respective counts
# categories = category_counts.index
# counts = category_counts.values

# # Create a bar plot 
# plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
# plt.bar(categories, counts)

# # Optionally, you can set labels for the x-axis and y-axis
# plt.xlabel('Arrival_City')
# plt.ylabel('Count')

# # Optionally, you can rotate the x-axis labels for better readability
# plt.xticks(rotation=45)

# # Show the plot
# plt.show()


# In[64]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Aircraft_Type'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Aircraft_Type')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[65]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Month_of_Travel'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Month_of_Travel')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[66]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Demand'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Demand')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[67]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Weather_Conditions'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Weather_Conditions')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[68]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have already loaded your DataFrame df

# Count the occurrences of each unique category
category_counts = df['Holiday_Season'].value_counts()

# Extract the category names and their respective counts
categories = category_counts.index
counts = category_counts.values

# Create a bar plot
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(categories, counts)

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Holiday_Season')
plt.ylabel('Count')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# # FEATURE VS TARGET EDA

# In[69]:


plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.bar(df['Departure_City'], df['Flight_Price'], label='Departure_City vs Flight_Price ')

# Optionally, you can set labels for the x-axis and y-axis
plt.xlabel('Departure_City')
plt.ylabel('Flight_Price')

# Optionally, you can rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.legend(loc='upper right')  # You can choose a different location

plt.show() 


# In[70]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Airline')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Airline vs Flight Price')
plt.xlabel('Airline')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[71]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Day_of_Week')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Day_of_Week vs Flight Price')
plt.xlabel('Day_of_Week')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Aircraft_Type')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Aircraft_Type vs Flight Price')
plt.xlabel('Aircraft_Type')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[73]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Month_of_Travel')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Month_of_Travel vs Flight Price')
plt.xlabel('Month_of_Travel')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[74]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Holiday_Season')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Holiday_Season vs Flight Price')
plt.xlabel('Holiday_Season')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[75]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Demand')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Demand vs Flight Price')
plt.xlabel('Demand')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[76]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Weather_Conditions')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Weather_Conditions vs Flight Price')
plt.xlabel('Weather_Conditions')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# In[77]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Group the data by 'Airline' and calculate the mean Flight_Price for each airline
airline_flight_price = df.groupby('Promotion_Type')['Flight_Price'].mean()

# Sort the values in descending order (for example, to show the highest average prices at the top)
airline_flight_price = airline_flight_price.sort_values(ascending=True)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
airline_flight_price.plot(kind='bar', edgecolor='k')
plt.title('Promotion_Type vs Flight Price')
plt.xlabel('Promotion_Type')
plt.ylabel('Average Flight Price')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()


# # import pandas as pd
# import matplotlib.pyplot as plt
# 
# # Assuming 'df' is your DataFrame
# 
# # Group the data by 'Airline' and calculate the mean Flight_Price for each airline
# airline_flight_price = df.groupby('Number_of_Stops')['Flight_Price'].mean()
# 
# # Sort the values in descending order (for example, to show the highest average prices at the top)
# airline_flight_price = airline_flight_price.sort_values(ascending=True)
# 
# # Create a bar plot
# plt.figure(figsize=(10, 6))  # Set the figure size
# airline_flight_price.plot(kind='bar', edgecolor='k')
# plt.title('Number_of_Stops vs Flight Price')
# plt.xlabel('Number_of_Stops')
# plt.ylabel('Average Flight Price')
# plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
# plt.show()
# 

# # cont vs cont

# In[78]:


df


# In[79]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Distance'], df['Flight_Price'], marker='o', color='green', label='Data Points')

# Set labels and title
plt.xlabel('Distance')
plt.ylabel('Flight_Price')
plt.title('Scatter Plot of Flight Price vs. distance')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[80]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Departure_Time'], df['Flight_Price'], marker='o', color='green', label='Data Points')

# Set labels and title
plt.xlabel('Departure_Time')
plt.ylabel('Flight_Price')
plt.title('Scatter Plot of Flight Price vs.Departure_Time')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[81]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Arrival_Time'], df['Flight_Price'], marker='o', color='green', label='Data Points')

# Set labels and title
plt.xlabel('Arrival_Time')
plt.ylabel('Flight_Price')
plt.title('Scatter Plot of Flight Price vs.Arrival_Time')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[82]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Duration'], df['Flight_Price'], marker='o', color='green', label='Data Points')

# Set labels and title
plt.xlabel('Duration')
plt.ylabel('Flight_Price')
plt.title('Scatter Plot of Flight Price vs.Duration')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[ ]:





# In[83]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Passenger_Count'], df['Flight_Price'], marker='o', color='green', label='Data Points')

# Set labels and title
plt.xlabel('Passenger_Count')
plt.ylabel('Flight_Price')
plt.title('Scatter Plot of Flight Price vs.Passenger_Count')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[84]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Fuel_Price'], df['Flight_Price'], marker='o', color='green', label='Data Points')

# Set labels and title
plt.xlabel('Fuel_Price')
plt.ylabel('Flight_Price')
plt.title('Scatter Plot of Flight Price vs.')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[85]:


df.drop(['Flight_ID', 'Departure_Time', 'Arrival_Time'], axis=1, inplace=True) # unwanted columns to drop


# In[86]:


df


# # ENCODING

# In[87]:


from sklearn.preprocessing import LabelEncoder
# Define the columns to be encoded with label encoding
ordinal_cols = ['Day_of_Week', 'Month_of_Travel', 'Holiday_Season', 'Demand', 'Promotion_Type']

# Apply ordinal/Label encoding
label_encoder = LabelEncoder()
for col in ordinal_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[88]:


df.head(10)


# In[89]:


# # Remove leading/trailing whitespaces from column names
# df.columns = df.columns.str.strip()

# # Define the columns to be encoded with one-hot encoding
# categorical_cols = ['Airline', 'Aircraft_Type', 'Weather_Conditions']

# # Apply one-hot encoding
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# In[ ]:





# In[90]:


df


# In[91]:


#Define the columns to be encoded with one-hot encoding
categorical_cols = ['Airline', 'Aircraft_Type', 'Weather_Conditions']
# Apply one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# In[92]:


df


# In[93]:


# Define the list of columns to convert to integers
int_columns = ['Airline_Airline B','Airline_Airline C','Aircraft_Type_Airbus A380','Aircraft_Type_Boeing 737','Aircraft_Type_Boeing 777','Aircraft_Type_Boeing 787','Weather_Conditions_Cloudy','Weather_Conditions_Rain','Weather_Conditions_Snow']

# Convert the selected columns to integers
df[int_columns] = df[int_columns].astype(int)


# In[94]:


# Perform leave-one-out encoding for card_type, card_number, and tid
from category_encoders import LeaveOneOutEncoder

looe_encoder = LeaveOneOutEncoder(cols=['Departure_City', 'Arrival_City'])
df= looe_encoder.fit_transform(df,df['Flight_Price'])


# In[95]:


pd.set_option('display.max_columns', 500)


# In[96]:


df


# # DATA SPLITTING

# In[97]:


import numpy as np 

from sklearn.model_selection import train_test_split


# In[98]:


X = df.drop('Flight_Price', axis=1)
y = df['Flight_Price']


# In[99]:


X


# In[100]:


y


# In[ ]:





# # SCALLING

# In[101]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X= scaler.fit_transform(X)
X


# # TRAIN TEST SPLIT

# In[102]:


from sklearn.model_selection import train_test_split
# test and train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:





# # model fitting

# In[103]:


get_ipython().system('pip install xgboost')


# In[104]:


# Import necessary libraries
import xgboost as xgb
from xgboost import XGBRegressor  # Use XGBRegressor for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[105]:


lr_model =XGBRegressor() 


# In[106]:


lr_model.fit(X_train, y_train,)


# In[107]:


y_predict = lr_model.predict(X_test)


# In[108]:


mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)


# In[109]:


mse ,r2


# In[110]:


from sklearn.model_selection import cross_val_score
import xgboost as xgb


# In[111]:


cross_val_xgb=[]
for lr in [0.01,0.05,0.08,0.1,0.2,0.25,0.3]:
  xgb_regressor= xgb.XGBRegressor(learning_rate = lr,n_estimators=100)
  xgb_regressor.fit(X_train,y_train)
  print("Learning rate : ", lr,"cross_val_score:", cross_val_score(xgb_regressor,X_train,y_train,cv = 15).mean())
  cross_val_xgb.append(cross_val_score(xgb_regressor,X_train,y_train,cv = 15).mean())


# In[112]:


from sklearn.model_selection import cross_val_score
import xgboost as xgb
xgb_regressor= xgb.XGBRegressor(learning_rate =0.1,n_estimators=100)
xgb_regressor.fit(X_train,y_train)
print("Learning rate : ",0.1,"cross_val_score:", cross_val_score(xgb_regressor,X_train,y_train,cv = 15).mean())
cross_val_xgb.append(cross_val_score(xgb_regressor,X_train,y_train,cv = 15).mean())


# In[113]:


cross_val_xgb_regressor=max(cross_val_xgb)
print("The best Learning rate is 0.1 and Cross_val_score is:",cross_val_xgb_regressor)


# In[114]:


# Make predictions
y_pred = xgb_regressor.predict(X_test)


# In[115]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)


# In[116]:


xgb_regressor.feature_importances_


# In[117]:


sorted_idx = xgb_regressor.feature_importances_.argsort()
plt.figure(figsize=(10,5))
plt.barh(df.columns[sorted_idx], xgb_regressor.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Feature Importance")
plt.show()


# In[118]:


test_data = pd.read_csv('test.csv')


# In[119]:


test_data


# In[120]:


test_data.isnull().sum()


# In[121]:


# Calculate the mode of the "Airline" column
mode_value = test_data['Airline'].mode()[0]

# Fill missing values in the "Airline" column with the mode
test_data['Airline'].fillna(mode_value, inplace=True)


# In[122]:


# Calculate the mode of the "Departure_City'" column
mode_value = test_data['Departure_City'].mode()[0]

# Fill missing values in the "Departure_City'" column with the mode
test_data['Departure_City'].fillna(mode_value, inplace=True)


# In[123]:


# Calculate the mode of the "Arrival_City" column
mode_value = test_data['Arrival_City'].mode()[0]

# Fill missing values in the "Arrival_City" column with the mode
test_data['Arrival_City'].fillna(mode_value, inplace=True)


# In[124]:


# Calculate the median of the "Distance" column
median_value = test_data['Distance'].median()

# Fill missing values in the "Distance" column with the mode
test_data['Distance'].fillna(median_value, inplace=True)


# In[125]:


# Calculate the mode of the "Airline" column
mode_value = test_data['Aircraft_Type'].mode()[0]

# Fill missing values in the "Airline" column with the mode
test_data['Aircraft_Type'].fillna(mode_value, inplace=True)


# In[126]:


# Calculate the mode of the "Day_of_Week" column
mode_value = test_data['Day_of_Week'].mode()[0]

# Fill missing values in the "Day_of_Week" column with the mode
test_data['Day_of_Week'].fillna(mode_value, inplace=True)


# In[127]:


# Calculate the mode of the "Month_of_Travel" column
mode_value = test_data['Month_of_Travel'].mode()[0]

# Fill missing values in the "Month_of_Travel" column with the mode
test_data['Month_of_Travel'].fillna(mode_value, inplace=True)


# In[128]:


# Calculate the mode of the "Holiday_Season" column
mode_value = test_data['Holiday_Season'].mode()[0]

# Fill missing values in the "Holiday_Season" column with the mode
test_data['Holiday_Season'].fillna(mode_value, inplace=True)


# In[129]:


# Calculate the mode of the "Demand" column
mode_value = test_data['Demand'].mode()[0]

# Fill missing values in the "Demand" column with the mode
test_data['Demand'].fillna(mode_value, inplace=True)


# In[130]:


# Calculate the mode of the "Weather_Conditions" column
mode_value = test_data['Weather_Conditions'].mode()[0]

# Fill missing values in the "Weather_Conditions" column with the mode
test_data['Weather_Conditions'].fillna(mode_value, inplace=True)


# In[131]:


# Calculate the mode of the "'Promotion_Type'" column
mode_value = test_data['Promotion_Type'].mode()[0]

# Fill missing values in the "'Promotion_Type'" column with the mode
test_data['Promotion_Type'].fillna(mode_value, inplace=True)


# In[132]:


# Calculate the median of the "Fuel_Price" column
median_value = test_data['Fuel_Price'].median()

# Fill missing values in the "Fuel_Price" column with the mode
test_data['Fuel_Price'].fillna(median_value, inplace=True)


# In[133]:


test_data.isnull().sum()


# In[134]:


test_data.dtypes


# In[135]:


test_data['Flight_ID'].value_counts()


# In[136]:


test_data['Airline'].value_counts()


# In[137]:


test_data['Departure_City'].value_counts()


# In[138]:


test_data['Arrival_City'].value_counts()


# In[139]:


test_data['Distance'].value_counts()


# In[140]:


test_data['Departure_Time'].value_counts()


# In[141]:


test_data['Arrival_Time'].value_counts()


# In[142]:


test_data['Duration'].value_counts()


# In[143]:


test_data['Aircraft_Type'].value_counts()


# In[144]:


test_data['Number_of_Stops'].value_counts()


# In[145]:


test_data['Day_of_Week'].value_counts()


# In[146]:


test_data['Month_of_Travel'].value_counts()


# In[147]:


test_data['Holiday_Season'].value_counts()


# In[148]:


test_data['Demand'].value_counts()


# In[149]:


test_data['Weather_Conditions'].value_counts()


# In[150]:


test_data['Passenger_Count'].value_counts()


# In[151]:


test_data['Promotion_Type'].value_counts()


# In[152]:


test_data['Fuel_Price'].value_counts()


# In[153]:


test_data.drop_duplicates()


# In[154]:


#Define the columns to be encoded with one-hot encoding
categorical_cols = ['Airline', 'Aircraft_Type', 'Weather_Conditions']
# Apply one-hot encoding
test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)


# In[155]:


# Define the list of columns to convert to integers
int_columns = ['Airline_Airline B','Airline_Airline C','Aircraft_Type_Airbus A380','Aircraft_Type_Boeing 737','Aircraft_Type_Boeing 777','Aircraft_Type_Boeing 787','Weather_Conditions_Cloudy','Weather_Conditions_Rain','Weather_Conditions_Snow']

# Convert the selected columns to integers
test_data[int_columns] = test_data[int_columns].astype(int)


# In[156]:


from sklearn.preprocessing import LabelEncoder
# Define the columns to be encoded with label encoding
ordinal_cols = ['Day_of_Week', 'Month_of_Travel', 'Holiday_Season', 'Demand', 'Promotion_Type']

# Apply ordinal/Label encoding
label_encoder = LabelEncoder()
for col in ordinal_cols:
    test_data[col] = label_encoder.fit_transform(test_data[col])


# In[157]:


test_data


# In[158]:


from category_encoders import LeaveOneOutEncoder
# Perform leave-one-out encoding for card_type, card_number, and tid
looe_encoder = LeaveOneOutEncoder(cols=['Departure_City', 'Arrival_City'])
test_data = looe_encoder.fit_transform(test_data, test_data['Distance'])


# In[159]:


test_data


# In[160]:


# Correct way (using the 'labels' parameter)
columns_to_drop = ['Flight_ID', 'Departure_Time', 'Arrival_Time']
test_data = test_data.drop(labels=columns_to_drop, axis=1)


# In[161]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[162]:


test_data


# In[163]:


dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)


# In[164]:


dtrain


# In[165]:


test_data = xgb.DMatrix(data=test_data, enable_categorical=True)


# In[166]:


import xgboost as xgb

# Assuming you already have dtrain and dtest defined from previous steps

# Specify the XGBoost parameters such as 'max_depth', 'eta', etc.
params = {
    'max_depth': 5,
    'eta': 0.2,
    'objective': 'reg:squarederror',  # Specify the appropriate objective for your problem
}

# Perform cross-validation with 10 folds
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,  # You can adjust the number of boosting rounds
    nfold=10,  # Number of cross-validation folds
    metrics={'rmse'},  # Evaluation metric (Root Mean Squared Error)
    early_stopping_rounds=10,  # Optional: Early stopping rounds
    seed=42  # Optional: Seed for reproducibility
)
print(cv_results)
# Get the best number of boosting rounds
best_num_boost_rounds = cv_results['test-rmse-mean'].idxmin()

# Re-train the model with the best number of boosting rounds
xgb_reg = xgb.train(params, dtrain, num_boost_round=best_num_boost_rounds)

# Make predictions on the test set
y_pred = xgb_reg.predict(test_data)


# In[167]:


def custom_round(x):
    round_value=round(x*100)
    return round_value/100


# In[168]:


y_pred1=np.vectorize(custom_round)(y_pred)


# In[169]:


submission = pd.read_csv("test.csv")

Predicted_flight_price=pd.DataFrame({'Flight_ID':submission ['Flight_ID'],'Flight_Price':y_pred1})


# In[170]:


Predicted_flight_price.to_csv('Submission.csv', index = False)


# In[171]:


Predicted_flight_price


# In[ ]:





# In[ ]:




