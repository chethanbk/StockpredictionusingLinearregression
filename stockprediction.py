import numpy as np   #Linear algera Library
import pandas as pd
import matplotlib.pyplot as plt  #to plot graphs
import seaborn as sns  #to plot graphs
from sklearn.linear_model import LinearRegression   #for linear regression model
sns.set()  #setting seaborn as default
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
import math
import warnings

warnings.filterwarnings('ignore')

data=pd.read_csv(r"NSE-Tata-Global-Beverages-Limited.csv")   #reads the input data
data.head()   #displays the first five rows
data.info()
data.describe(include ='all')   #parameter include=all will display NaN values as well
data.isnull().sum() # No null values
data.head()
sns.pairplot(data)
plt.show()
x=data[['High','Low','Last','Open','Total Trade Quantity','Turnover (Lacs)']].values   #input
y=data[['Close']].values   #output
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)
lm=LinearRegression()
lm.fit(x_train,y_train)
print("lm.coef_::",lm.coef_)
lm.score(x_train,y_train)
print("lm Score::", lm.score)
predictions = lm.predict(x_test)

r2_score(y_test, predictions)
print("r2 score::", r2_score)
dframe=pd.DataFrame({'actual':y_test.flatten(),'Predicted':predictions.flatten()})
print(dframe.head(15))

graph =dframe.head(10)
graph.plot(kind='bar')
plt.title('Actual vs Predicted')
plt.ylabel('Closing price')

fig = plt.figure()
plt.scatter(y_test,predictions)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)
plt.ylabel('Predicted', fontsize=20)
plt.show()
# sns.regplot (y_test, predictions)
# plt.title('Actual versus Prediction ')
# plt.xlabel('Actual', fontsize=20)
# plt.ylabel('Predicted', fontsize=20)
print('Mean Abs value:' ,metrics.mean_absolute_error(y_test,predictions))
print('Mean squared value:',metrics.mean_squared_error(y_test,predictions))
print('root mean squared error value:',math.sqrt(metrics.mean_squared_error(y_test,predictions)))