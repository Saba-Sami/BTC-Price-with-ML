from pandas import *
from numpy import * 
from matplotlib.pyplot import *


df = read_csv("C:/codes/python-codes/machine learning/samples/bitcoinprice.csv")
df['timeLow']=to_datetime(df['timeLow'] , unit='ms')
df['timeHigh']=to_datetime(df['timeHigh'] , unit='ms')
df['timeOpen']=to_datetime(df['timeOpen'] , unit='ms')
df['timeClose']=to_datetime(df['timeClose'] , unit='ms')
df.head(10)


scatter(df.timeOpen , df.priceClose , color="blue")
show()



cdf= df [[ 'timeOpen', 'timeClose'  , 'priceHigh' , 'priceLow' , 'priceClose']]
cdf.loc[:, 'Opendate_num'] = (df['timeOpen'] - df['timeOpen'].min()).dt.days
cdf.loc[:, 'Closedate_num'] = ((df['timeClose'] - df['timeClose'].min()).dt.days)+1
cdf.tail(10)

mdf = cdf[[ 'Opendate_num', 'Closedate_num'  , 'priceHigh' , 'priceLow' , 'priceClose']]
mdf.head(10)


mask = random.rand(len(mdf)) < 0.8
train = mdf[ mask ]
test = mdf[~mask ]

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = asanyarray(train[[ 'Opendate_num', 'Closedate_num'  , 'priceHigh' , 'priceLow' ]])
train_y = asanyarray(train[['priceClose']])
regr.fit(train_x , train_y)
print( f"priceClose =  ({round(regr.coef_[0][0] , 3)})Opendate_num + ({round(regr.coef_[0][1],3)})Closedate_num + ({round(regr.coef_[0][2],3)})priceHigh + ({round(regr.coef_[0][3],3)})priceLow + ({round(regr.intercept_[0],3)})")


from sklearn.metrics import r2_score
test_x = asanyarray(test[[ 'Opendate_num', 'Closedate_num'  , 'priceHigh' , 'priceLow' ]])
test_y = asanyarray(test[['priceClose']])
yhat = regr.predict(test_x)
r2_score(yhat  , test_y)


my_test = [[2502 , 2503 , 107772 , 106449]]
bitcoinprice = regr.predict(my_test )
print(f"Bitcon price is : {bitcoinprice[0][0]}")

