#!/usr/bin/env python
# coding: utf-8

# In[2]:


from binance.client import Client
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import bar_chart_race as bcr
import math
import seaborn as sns
import bar_chart_race as bcr


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
lr = LinearRegression(n_jobs=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(loss = 'squared_error', criterion = 'friedman_mse')

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.layers import Dense
LSTM_model = Sequential()


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
def Extract_Data(length):
    '''
    This Function return a DataFrame about Crypto
    ----------------------------------------------
    Element: legnth (int value)
    '''
    client = Client()
    
    date = datetime.datetime.today()
    date = date.date()
    
    dataframe = pd.date_range(start='01/01/2020' , end=date, freq="1D").to_frame()
    dataframe.index.name = 'Open time'
    dataframe.drop([0], axis=1, inplace=True)
    
    coin_list = ['BTCBUSD', 'ETHBUSD', 'BNBBUSD'] #List of Cryptocurrency 
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore']
    
    try:
        for i in range(length):
            temp = pd.DataFrame(client.get_historical_klines(coin_list[i], Client.KLINE_INTERVAL_1DAY,'1 Jan 2020'), columns=columns) #fetch weekly klines since it listed
            temp.index = pd.to_datetime(temp['Open time'], unit='ms')
            temp.drop(['Open time','Close time','Quote asset volume','Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore'], axis=1, inplace=True)
            print('Extract: ',i+1,')', coin_list[i], '\n')
            
            for j in range(len(temp.columns)): #Rename the columns
                temp.rename(columns={temp.columns[j] : coin_list[i]+'_'+temp.columns[j]}, inplace=True) 
            dataframe = dataframe.join(temp)
            #print(temp)
    except:
        print('Length of coin_list is --->', len(coin_list))
    return dataframe
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def Select_Target(dataframe , name_of_col):
    '''
    This function create a Target!
    And drop null rows of the dataframe
    ----------------------------------------------
    Element: dataframe (Is a dataframe)
    '''
    print('Choice the Target!')
    for i in range(len(dataframe.columns)):
        print(i,': ', dataframe.columns[i])
    
    target_columns = dataframe[name_of_col].shift(-1) #Select Target
    dataframe.insert(0, 'Target_'+name_of_col, target_columns) #First position
    dataframe.dropna(inplace=True) #Drop the null rows of dataframe 
    
    #print(dataframe.columns)
    #print(dataframe)
    #return dataframe
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def set_null_value(dataframe):
    for i in range(len(dataframe.columns)):
        #print(i,':', df.columns[i], 'have ', df[df.columns[i]].isnull().sum(), 'null values')
        if (dataframe[dataframe.columns[i]].isnull().sum() > 0):
            dataframe[dataframe.columns[i]].ffill(axis = 0, inplace=True)
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def Convert_Type(dataframe): 
    '''
    This function converte the type of columns.
    From object to float.
    ----------------------------------------------
    Element: dataframe (Is a dataframe)
    '''
    for i in dataframe.columns:
        if ((dataframe[i].dtypes == 'Object') or (dataframe[i].dtypes == 'object') or (dataframe[i].dtypes == 'O') or (dataframe[i].dtypes == 'o')):
            dataframe[i] = dataframe[i].astype(float)
    dataframe.round(2)
    #return dataframe
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def train_test_split_df(dataframe, t_size, shuff):
    '''
    This function split the data
    ----------------------------------------------
    Element: dataframe (Is a dataframe)
    Element: t_size (Is float number <0.2 , 0.3> and split the data) 
    Element: shuff (Is Bool, If true the rows is random)
    '''
    
    X = dataframe.iloc[:,1:]
    Y = dataframe.iloc[:,0]
    
    #global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=t_size, shuffle=shuff)
    
    sum_train_test = len(y_train) + len(y_test)
    train_set = int(100 - t_size*100)
    test_set = int(t_size*100)
    
    fig = plt.figure(figsize=(15,5), dpi=300)
    plt.title('{}'.format(Y.name), color='White', fontsize=20, x=0.19, y=0.85)
    plt.figtext(x=0.168, y=0.7, s='Train {}% of the Data-Set'.format(train_set), color='blue', fontsize=15)
    plt.figtext(x=0.168, y=0.63, s='Test {}% of the Data-Set'.format(test_set), color='orange', fontsize=15)
    plt.plot(np.arange(0,len(y_train),1), y_train, label='Training set', color='blue')
    plt.plot(np.arange(len(y_train) ,sum_train_test, 1), y_test, label='Test set', color='orange')
    plt.xlabel('Period', fontsize=20, color='Gold')
    plt.ylabel('Price', fontsize=20, color='Gold')
    plt.tick_params(axis='x', colors='White', labelsize=15)
    plt.tick_params(axis='y', colors='White', labelsize=15)
    #plt.grid(True)
    plt.axis('on')
    #plt.legend()
    
    img_tts = plt.savefig('../Application_Programming_Interface/static/images/img_web/train_test_split_img.png', transparent=True) #Download Image
    plt.show()
    return X_train, X_test, y_train, y_test
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def Eval_Of_Model_Forecast(model, X_train, X_test, y_train, y_test, name_of_img, model_type):
    
    Xtrain = X_train.copy()
    Xtest = X_test.copy()
    ytrain = y_train.copy()
    ytest = y_test.copy()
    ytest_index = ytest.index
    columns = Xtrain.columns
    idx_Train = Xtrain.index
    idx_Test = Xtest.index
    actual_price = y_test.copy()
    
    #dataframe = pd.DataFrame()
    #Metric = pd.DataFrame()
    
    list_of_prediction = []
    
    
    if (model_type == 'ML'): #Machine Learning
        while len(Xtest) > 0:
            model.fit(Xtrain,ytrain)
        
            if len(Xtest) > 7:
                prediction = model.predict(Xtest.iloc[0:7]).tolist()
                length_of_predict = len(prediction)
                list_of_prediction = list_of_prediction + prediction
            
                Xtrain = pd.concat([Xtrain,Xtest.iloc[0:7]]) #Insert rows on the dataframe
                ytrain = pd.concat([ytrain,ytest.iloc[0:7]])
        
                Xtest.drop(Xtest.index[range(7)], inplace=True) #Drop Rows from Test
                ytest.drop(ytest.index[range(7)], inplace=True)
                Xtrain.drop(Xtrain.index[range(7)], inplace=True) #Drop Rows from Train
                ytrain.drop(ytrain.index[range(7)], inplace=True)
            
            else:
                prediction = model.predict(Xtest).tolist()
                list_of_prediction = list_of_prediction + prediction
            
                Xtest.drop(Xtest.index,inplace=True) 
                ytest.drop(ytest.index,inplace=True) 
    
        dataframe = pd.DataFrame({'Date' : pd.to_datetime(ytest_index),
                                  'Forecast' : list_of_prediction,
                                  'Actual' : actual_price},
                                 index=ytest_index)
        print(dataframe)
    
        # Metrics
        model_train = model.predict(Xtrain)
        model_MSE_train = mean_squared_error(ytrain, model_train) #Train
        model_RMSE_train = math.sqrt(model_MSE_train)

        model_MSE_test = mean_squared_error(dataframe.Actual, dataframe.Forecast) #Test
        model_RMSE_test = math.sqrt(model_MSE_test)   
        
        Metric = pd.DataFrame({'RMSE\n'+'Train': model_RMSE_train,
                           'RMSE\n'+'Test' : model_RMSE_test},index=[1])
        Metric.astype(float)
    # Metrics
    
    
    
    # LSTM
    if model_type == 'NN': #Neural Network
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(Xtrain.shape[1],1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        
        while len(Xtest) > 0:
            
            Xtrain = np.reshape(np.array(Xtrain), (Xtrain.shape[0],Xtrain.shape[1], 1)) #Reshape 3d array
            Xtest = np.reshape(np.array(Xtest), (Xtest.shape[0],Xtest.shape[1], 1)) #Reshape 3d array
            model.fit(Xtrain,ytrain) #Fit the model
            
            if len(Xtest) > 7:
                prediction = model.predict(Xtest[0:7]).flatten().tolist()
                length_of_predict = len(prediction)
                list_of_prediction = list_of_prediction + prediction
            
                Xtrain = np.reshape(Xtrain, (Xtrain.shape[0],Xtrain.shape[1])) #Reshape 2d array
                Xtest = np.reshape(Xtest, (Xtest.shape[0],Xtest.shape[1])) #Reshape 2d array
                Xtrain = pd.DataFrame(Xtrain, columns=columns, index=idx_Train) #Create DataFrame Train
                Xtest = pd.DataFrame(Xtest, columns=columns, index=idx_Test) #Create DataFrame Test
            
                Xtrain = pd.concat([Xtrain,Xtest.iloc[0:7]]) #Insert rows on the dataframe
                ytrain = pd.concat([ytrain,ytest.iloc[0:7]]) #Insert rows on the dataframe
        
                Xtest.drop(Xtest.index[range(7)], inplace=True) #Drop Rows from Test
                ytest.drop(ytest.index[range(7)], inplace=True) #Drop Rows from Test
                Xtrain.drop(Xtrain.index[range(7)], inplace=True) #Drop Rows from Train
                ytrain.drop(ytrain.index[range(7)], inplace=True) #Drop Rows from Train
            
                idx_Train = Xtrain.index
                idx_Test = Xtest.index

            
            else:
                prediction = model.predict(Xtest).flatten().tolist()
                list_of_prediction = list_of_prediction + prediction
                Xtrain = np.reshape(Xtrain, (Xtrain.shape[0],Xtrain.shape[1])) #Reshape 2d array
                Xtest = np.reshape(Xtest, (Xtest.shape[0],Xtest.shape[1])) #Reshape 2d array
                Xtrain = pd.DataFrame(Xtrain, columns=columns, index=idx_Train) #Create DataFrame Train
                Xtest = pd.DataFrame(Xtest, columns=columns, index=idx_Test) #Create DataFrame Test
            
                Xtest.drop(Xtest.index,inplace=True) 
                ytest.drop(ytest.index,inplace=True) 
     
        dataframe = pd.DataFrame({'Date' : pd.to_datetime(ytest_index),
                                  'Forecast' : list_of_prediction,
                                  'Actual' : actual_price},
                                 index=ytest_index)
        print(dataframe)

        # Metrics
        Xtrain = np.reshape(np.array(Xtrain), (Xtrain.shape[0],Xtrain.shape[1], 1)) #Reshape 3d array
        model_train = model.predict(Xtrain)
        model_MSE_train = mean_squared_error(ytrain, model_train) #Train
        model_RMSE_train = math.sqrt(model_MSE_train)
        print('RMSE_Train: ', model_RMSE_train)
    
        model_MSE_test = mean_squared_error(dataframe.Actual, dataframe.Forecast) #Test
        model_RMSE_test = math.sqrt(model_MSE_test)   
       
        Metric = pd.DataFrame({'RMSE\n'+'Train': model_RMSE_train,
                               'RMSE\n'+'Test' : model_RMSE_test},index=[1])
        Metric.astype(float)
        # Metrics
    #LSTM
    
    
    #Create Plots
    fig = gridspec.GridSpec(2, 2)
    pl.figure(figsize=(17, 10), tight_layout=True, dpi=300)
    
    #First Plot
    axis1 = pl.subplot(fig[0,:])
    axis1.set_title('Forecasting of the Test-Set', color='Gold', fontsize=20,  x=0.5)
    axis1.plot(dataframe.Date, dataframe.Actual, label='Actual', color='Blue', linewidth=0.8)
    axis1.plot(dataframe.Date, dataframe.Forecast, label='Prediction', color='Red', linewidth=1.5)
    axis1.tick_params(axis='x', colors='White',direction="in",width=10, length=12, labelrotation=-60, labelsize=15)
    axis1.tick_params(axis='y', colors='White',direction="in",width=10, length=12, labelrotation=20, labelsize=15)
    axis1.grid(linewidth = 0.5, color='Green')
    axis1.spines['top'].set_visible(False)
    axis1.spines['right'].set_visible(False)
    axis1.spines['left'].set_visible('White')
    axis1.spines['bottom'].set_color('White')
    axis1.axis('on')
    pl.figtext(x=0.92, y=0.925, s='Forecast', color='Blue', fontsize=15)
    pl.figtext(x=0.92, y=0.88, s='Actual', color='Red', fontsize=15)
    pl.ylabel('Price', fontsize=20, color='Gold')
    
    #Second Plot
    axis2 = pl.subplot(fig[1,0])
    axis2.bar(Metric.columns[0], Metric.iloc[:,0],width = 0.5, color='Blue')
    axis2.bar(Metric.columns[1], Metric.iloc[:,1],width = 0.5, color='Red')
    axis2.tick_params(axis='x', colors='White',direction="in",width=10, length=12, labelrotation=-60, labelsize=15)
    axis2.tick_params(axis='y', colors='White',direction="in",width=10, length=12, labelrotation=20, labelsize=15)
    axis2.grid(axis='y')
    axis2.spines['top'].set_visible(False)
    axis2.spines['right'].set_visible(False)
    axis2.spines['left'].set_visible(False)
    axis2.spines['bottom'].set_color('White')
    pl.ylabel('Points', fontsize=20, color='Gold')
    
    #Third Plot
    axis3 = pl.subplot(fig[1,1])
    axis3.barh(Metric.columns[0], Metric.iloc[:,0],height = 0.5, color='Blue')
    axis3.barh(Metric.columns[1], Metric.iloc[:,1],height = 0.5, color='Red')
    axis3.tick_params(axis='x', colors='White',direction="in",width=10, length=12, labelrotation=-60, labelsize=15)
    axis3.tick_params(axis='y', colors='White',direction="in",width=10, length=12, labelrotation=20, labelsize=15)
    axis3.grid(axis='x')
    axis3.spines['top'].set_visible(False)
    axis3.spines['right'].set_visible(False)
    axis3.spines['left'].set_color('White')
    axis3.spines['bottom'].set_visible(False)
    
    img_lr = plt.savefig('../Application_Programming_Interface/static/images/img_web/Model'+name_of_img+'.png',  
                         bbox_inches='tight',
                         transparent=True)
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def forecasting(model1, model2, model3, model4, dataframe):
    model1_pred = []
    model2_pred = []
    model3_pred = []
    model4_pred = []
    
    dataframe_evaluation = dataframe.copy()
    X = dataframe_evaluation.iloc[:, 1:]
    y = dataframe_evaluation.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    ytest = y_test.copy()
    
    while len(X_test) > 0:
        model1.fit(X_train,y_train)
        model2.fit(X_train,y_train)
        model3.fit(X_train,y_train)
        model4.fit(X_train,y_train)
        
        if len(X_test) > 7:
            #-----------------------------------------------------
            prediction1 = model1.predict(X_test.iloc[0:7]).tolist()
            model1_pred = model1_pred + prediction1
            #-----------------------------------------------------
            prediction2 = model2.predict(X_test.iloc[0:7]).tolist()
            model2_pred = model2_pred + prediction2
            #-----------------------------------------------------
            prediction3 = model3.predict(X_test.iloc[0:7]).tolist()
            model3_pred = model3_pred + prediction3
            #-----------------------------------------------------
            prediction4 = model4.predict(X_test.iloc[0:7]).tolist()
            model4_pred = model4_pred + prediction4
            #-----------------------------------------------------
            
            X_train = pd.concat([X_train, X_test.iloc[0:7]]) #Insert rows on the dataframe
            y_train = pd.concat([y_train, y_test.iloc[0:7]])
        
            X_test.drop(X_test.index[range(7)], inplace=True) #Drop Rows from Test
            y_test.drop(y_test.index[range(7)], inplace=True)
            X_train.drop(X_train.index[range(7)], inplace=True) #Drop Rows from Train
            y_train.drop(y_train.index[range(7)], inplace=True)
            
        else:
            #-----------------------------------------------------
            prediction1 = model1.predict(X_test.iloc[0:7]).tolist()
            model1_pred = model1_pred + prediction1
            #-----------------------------------------------------
            prediction2 = model2.predict(X_test.iloc[0:7]).tolist()
            model2_pred = model2_pred + prediction2
            #-----------------------------------------------------
            prediction3 = model3.predict(X_test.iloc[0:7]).tolist()
            model3_pred = model3_pred + prediction3
            #-----------------------------------------------------
            prediction4 = model4.predict(X_test.iloc[0:7]).tolist()
            model4_pred = model4_pred + prediction4
            #-----------------------------------------------------
            X_test.drop(X_test.index,inplace=True) 
        
    dataframe1 = pd.DataFrame({'Forecast' : model1_pred,
                              'Actual' : ytest},
                              index=ytest.index)
    
    dataframe2 = pd.DataFrame({'Forecast' : model2_pred,
                              'Actual' : ytest},
                              index=ytest.index)
    
    dataframe3 = pd.DataFrame({'Forecast' : model3_pred,
                               'Actual' : ytest},
                              index=ytest.index)
    
    dataframe4 = pd.DataFrame({'Forecast' : model4_pred,
                               'Actual' : ytest},
                              index=ytest.index)
    
    mse1 = mean_squared_error(dataframe1.Actual, dataframe1.Forecast)
    mse2 = mean_squared_error(dataframe2.Actual, dataframe2.Forecast)
    mse3 = mean_squared_error(dataframe3.Actual, dataframe3.Forecast)
    mse4 = mean_squared_error(dataframe4.Actual, dataframe4.Forecast)
    
    rmse1 = math.sqrt(mse1)
    rmse2 = math.sqrt(mse2)
    rmse3 = math.sqrt(mse3)
    rmse4 = math.sqrt(mse4)
    
    #print(rmse1)
    #print(rmse2)
    #print(rmse3)
    #print(rmse4)
    
    data = {model1 : rmse1,
            model2 : rmse2,
            model3 : rmse3,
            model4 : rmse4}
    
    model_min_rmse = min(data, key=data.get)
    #print('\nThe wich has smaller RMSE is: ', model_min_rmse)
    
    New_X = dataframe_evaluation.iloc[-801:-1, 1:]
    New_y = dataframe_evaluation.iloc[-801:-1, 0]
    test = pd.DataFrame(dataframe_evaluation.iloc[-1,1:])
    test = test.T
    
    model_min_rmse.fit(New_X,New_y)
    prediction = model_min_rmse.predict(test)[0]
    
    #print('--------------------------------------------------\n       !!!...- FORECASTING -...!!!')
    #print('               ',name)
    # 
    #print('   ', test.index[0],'23:59:59   :   {:.2f}'.format(prediction))
    #print('--------------------------------------------------')
    print('test.index[0]: {:.2f}'.format(prediction))
    return prediction
#-------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------
def visualize_all_close_prices(dataframe):
    visualize_all_close_df = dataframe[['BTCBUSD_Close', 'ETHBUSD_Close', 'BNBBUSD_Close']]
    visualize_all_close_df.insert(0,'Date', dataframe.index)
    
    fig = gridspec.GridSpec(1,1)
    pl.figure(figsize=(17, 10), tight_layout=True, dpi=300)
    
    axis1 = pl.subplot(fig[0,:])
    plt.plot(visualize_all_close_df.Date, visualize_all_close_df.BTCBUSD_Close, color='Blue')
    plt.plot(visualize_all_close_df.Date, visualize_all_close_df.ETHBUSD_Close, color='Red')
    plt.plot(visualize_all_close_df.Date, visualize_all_close_df.BNBBUSD_Close, color='Green')
    plt.tick_params(axis='x', colors='White',direction="in",width=5, length=10, labelrotation=-60, labelsize=12)
    plt.tick_params(axis='y', colors='White',direction="in",width=5, length=10, labelrotation=20, labelsize=12)
    plt.ylabel('Price', fontsize=20, color='Gold')
    plt.figtext(x=0.5, y=0.96, s='BITCOIN', color='Blue', fontsize=23)
    plt.figtext(x=0.7, y=0.96, s='ETHERIUM', color='Red', fontsize=23)
    plt.figtext(x=0.9, y=0.96, s='BINANCE', color='Green', fontsize=23)
    
    axis1.spines['top'].set_visible(False)
    axis1.spines['right'].set_visible(False)
    axis1.spines['left'].set_color('Grey')
    axis1.spines['bottom'].set_color('Grey')

    img_lr = plt.savefig('../Application_Programming_Interface/static/images/img_web/visualize_all_close.png',  
                         bbox_inches='tight',
                         transparent=True)
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------
    
    
    
    
    


# In[ ]:




