from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import time
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

#Read dataset
data = pd.read_csv('abalone.data',sep=',', header=None,
                   names = ['Sex','Length','Diameter',
                             'Height','Whole Weight', 'Shucked Weight',
                                           'Viscera Weight','Shell Weight',
                                            'Rings'])


#Create dummy variables for sex types
data['Male'] = (data['Sex']=='M').astype(int)
data['Female'] = (data['Sex']=='F').astype(int)
data['Infant'] = (data['Sex']=='I').astype(int)
data = data.drop(['Sex'],axis=1)

#Remove invalid data
data = data[data['Height'] > 0]

#Split and prepare data
train_data, test_data = model_selection.train_test_split(data,train_size=0.7)
xtrain = train_data.drop(['Rings'],axis = 1)
ytrain = train_data['Rings']
xtest = test_data.drop(['Rings'],axis = 1)
ytest = test_data['Rings']

#Scale data for zero mean and unit variance
scaler = StandardScaler()
scaler.fit(xtrain)
scaled_xtrain = scaler.transform(xtrain)
scaled_xtest = scaler.transform(xtest)

file = open("test_data4","w+")

for neurons in [10,30,50,80,100]:
    for layers in [2,4,5,8,10,50,100]:
        for activation_func in ['identity','logistic','tanh','relu']:
            #Create multilayer perceptron regression model
            model = MLPRegressor(hidden_layer_sizes = [neurons]*layers,
                                 alpha = 0.00001,
                                 activation = activation_func,
                                 learning_rate_init = 0.001,
                                 max_iter = 100000)

            #Fit model
            model.fit(scaled_xtrain,ytrain)
        
            #Predict output
            ypred = model.predict(scaled_xtest)

            rounded_y = [round(elem) for elem in ypred]
            yr = abs(ypred-ytest)

            file.write("%8d,%8d,%12s,%10.5f\n" % (neurons,layers,activation_func,mean_absolute_error(ytest, ypred)))


'''
for alpha_val in [0.1,0.01,0.001,0.0001,0.00001,0.000001]:
    for learning_rate in [0.1,0.01,0.001,0.0001]:
            #Create multilayer perceptron regression model
            model = MLPRegressor(hidden_layer_sizes = [50]*8,
                                 alpha = alpha_val,
                                 activation = 'tanh',
                                 learning_rate_init = learning_rate,
                                 max_iter = 100000000)

            #Fit model
            start = time.time()
            model.fit(scaled_xtrain,ytrain)
            end = time.time()
            
            #Predict output
            ypred = model.predict(scaled_xtest)

            file.write("%10.6f,%10.6f,%5.4d,%10.5f,%10.5f\n" % (alpha_val,learning_rate,end-start,mean_absolute_error(ytest, ypred),r2_score(ytest, ypred)))
'''
