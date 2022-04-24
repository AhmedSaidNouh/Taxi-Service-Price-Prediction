from time import  time
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

##############################################################################

labelEnconer=LabelEncoder()

##############################################################################

#load data
taxi = pd.read_csv('taxi-rides.CSV')
weather = pd.read_csv('weather.CSV')

##############################################################################

weather.rename(columns = {'location':'destination'}, inplace = True)

#fix the time_stamp column
taxi['time_stamp'] = pd.to_datetime(taxi['time_stamp']/1000)
weather['time_stamp'] = (weather['time_stamp']//10000)*10000
weather['time_stamp'] = pd.to_datetime(weather['time_stamp'])

#remove duplicates
weather.drop_duplicates(subset=['destination','time_stamp'],inplace=True)

#merge two data
taxiFinalData = taxi.merge(weather, how='inner', on=['destination','time_stamp'])

#remove duplicated data from the final data
taxiFinalData.drop_duplicates(inplace=True)

#assign input (Features) and output (Target)
Y=taxiFinalData['price']
X=taxiFinalData.drop(['price','id','wind','time_stamp','rain','pressure','temp'],axis=1,inplace=False)

#Split the data into Training set and Testing set
X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

#encoding categorigal data
colums=('name','cab_type','destination','source','product_id')
for i in colums:
         X_train[i]=labelEnconer.fit_transform(X_train[i])
         X_test[i]=labelEnconer.fit_transform(X_test[i])


#Fill null values with the mean in training set
X_train.fillna(value=X_train.mean(), inplace=True)
y_train.fillna(value=y_train.mean(), inplace=True)

#Fill null values with the mean in testing set
X_test.fillna(value=X_test.mean(), inplace=True)
y_test.fillna(value=y_test.mean(), inplace=True)

#apply P-Value
X_p=sm.add_constant(X_train)
p_value=sm.OLS(y_train,X_p).fit()
print(p_value.summary())


###############################################################################

#Apply multi-Linear Regression on The data
prediction_model=linear_model.LinearRegression()
start=time()
prediction_model.fit(X_train,y_train)
print("time_training_linear:",time()-start)
y_prediction_Train=prediction_model.predict(X_train)
mse_train_error=metrics.mean_squared_error(y_train,y_prediction_Train)
print("mse_train_linear:",mse_train_error)

y_prediction_Test=prediction_model.predict(X_test)
mse_test_error=metrics.mean_squared_error(y_test,y_prediction_Test)
print("mse_test_linear:",mse_test_error)


#Plot model
sns.set_theme(color_codes=True)
sns.regplot(x=y_test, y=y_prediction_Test, ci=None, color="b")
plt.show()
print(r2_score(y_test,y_prediction_Test))

print("-----------------------------------------------------------------------")

###############################################################################

#Apply Polynomial Regression on The data

poly_features = PolynomialFeatures(degree=4)

X_train=poly_features.fit_transform(X_train)
X_test=poly_features.fit_transform(X_test)

poly_model=linear_model.LinearRegression()
start=time()
poly_model.fit(X_train,y_train)
print("time_training_poly:",time()-start)
y_train_pre=poly_model.predict(X_train)

y_test_pre=poly_model.predict(X_test)

mse_train_error=metrics.mean_squared_error(y_train,y_train_pre)
print("mse_train_poly:",mse_train_error)

mse_test_error=metrics.mean_squared_error(y_test,y_test_pre)
print("mse_test_poly:",mse_test_error)

#Plot model
sns.set_theme(color_codes=True, style= "ticks")
sns.regplot(x=y_test, y=y_test_pre, ci=None, color="b" , order=3)
plt.show()
print(r2_score(y_test,y_test_pre))
########################### The End Of Milestone 1 ###########################
