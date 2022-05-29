"""

*** This Code Works on "univariate" (non)linear regression ***
*** It uses Gradient Descent Algorithm ***

"""
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
###############################################################################


###############################################################################

#-- Normalize -----------------------------------------------------------------
'''
    "Mean_Normalize" normalize the range of features
        by Mean Normalization Method
'''
def Mean_Normalize(a):
    return (a - np.mean(a))/(np.max(a) - np.min(a))
#------------------------------------------------------------------------------

#--Regression_GradientDescent--------------------------------------------------
'''
    "Regression_GradientDescent" run regression algorithm by Gradient Descent
        using MSE cost function
    it returns final theta values 
'''
def Regression_GradientDescent(x,y,order,epoch,leraning_rate,regular_rate = 0):
    
    #-- initialize theta values --   
    theta = np.random.rand(order+1)
    
    #-- DS size --
    m = len(x)
    
    #-- Get Different Powers of x values --
    #-- x_powers[i] = x^i for all samples --
    x_powers = X_Powers(x, order)   
    
    for k in range(epoch):
        theta_new = np.copy(theta)
        for i in range(len(theta)):            
            
            h = H(order=order, x=x ,x_powers = x_powers , theta = theta)
            e = h - y_train       
            theta_new[i] = theta_new[i] - (leraning_rate/m) * (np.sum(e*x_powers[i])+ 2 * regular_rate * theta_new[i]) 
        
        theta = np.copy(theta_new)
        
    return theta

#--Powers of X-----------------------------------------------------------------
'''
    "X_Powers" calculates different powers for x
    x_pw[i] = x^i for all samples
'''
def X_Powers(x,order):
    x_pw = {}    
    for i in range(0,order+1):        
        p = pow(x,i)
        p = np.float64(p)
        if i !=0:
            p = Mean_Normalize(p)
        x_pw[i] = p        
    return x_pw
#------------------------------------------------------------------------------

#--Hypothesis(x,y)-------------------------------------------------------------
'''
    "H" is a hypothesis function which 
        maps input variable(s) to a numeric output
    "H" is a univariate non-linear regression funtion
    h[i] = h(x_i)
'''
def H(order,x,x_powers,theta):
    h=theta[0]
    for i in range(1,order+1):
        x_pow_i = x_powers[i]        
        h = h + theta[i] * x_pow_i   
    
    return h
#------------------------------------------------------------------------------

#--Cost function for Gradient Descent------------------------------------------
'''
    "Cost_GD" calculates Mean Square Error 
'''
def Cost_GD(x,y,order,theta):
    x_test_pows = X_Powers(x,order)
    m = len(x)
    h = H(order,x,x_test_pows,theta)
    e = h - y
    cost = (1/(2*m))*np.sum(pow(e,2))
    return cost
#------------------------------------------------------------------------------
###############################################################################
    

###############################################################################
#-- Load DS -------------------------------------------------------------------
parent_dir = os.path.dirname(os.getcwd())
df = pd.read_csv(parent_dir+ '/dataset/ds.csv',sep=',')

#-- Show DS -------------------------------------------------------------------
fig = plt.figure(figsize=(5,5))
plt.scatter(x=df['x'], y=df['y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



#-- Shuffle -------------------------------------------------------------------
df = df.sample(frac=1)

#--Split DS t0 train and test -------------------------------------------------
#-- train = 70% and test = 30% --
df_copy = df.copy()
train_set = df_copy.sample(frac=0.7, random_state=0)
test_set = df_copy.drop(train_set.index)

#-- Get X and Y --
x_train = train_set.iloc[:,0].values
y_train = train_set.iloc[:,1].values

x_test = test_set.iloc[:,0].values
y_test = test_set.iloc[:,1].values

#-- Normalize Features --
x_train = Mean_Normalize(x_train)
x_test = Mean_Normalize(x_test)

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

###############################################################################

###############################################################################
#-- Regression using Gradient Descent -----------------------------------------
#-- Test different orders and different epochs --------------------------------
order_values = [2,3,5,7,10]
epoch_values = [1000,4000,10000]
colors =['red','green','black','blue','purple']

#-- errorvalues([i,j]) = error for order = i , epochs = j --
error_values = {}

fig, axs = plt.subplots(3, 5, figsize=(25,15)) 

for i in range(len(order_values)):
    
    order = order_values[i]    
    
    #-- log --
    print("Order= %d -----------------------------------"  %(order))       
        
    for j in range(len(epoch_values)):
        
        epoch = epoch_values[j]
        
        #-- log --
        print("\torder= %d"  %(epoch))
        
        #-- Plot DS --
        axs[j,i].scatter(x=X, y=y )
        
        #-- Run Gradient Descent and Get theta values --
        theta = Regression_GradientDescent(x=x_train, y=y_train, order=order, epoch=epoch, leraning_rate=0.01)
        
        #-- Use theta values to predict y for all data --
        x_powers = X_Powers(X, order=order)
        y_pred = H(order=order,x=X,theta = theta,x_powers = x_powers)

        #-- Plot Results --    
        axs[j,i].scatter(x=X , y = y_pred,c=str(colors[i]),s=8)         
        axs[j,i].set_title("order="+str(order)+" epochs=" + str(epoch))
                 
        #-- Get MSE error for test data --
        error = Cost_GD(x_test, y_test, order=order,theta = theta)
        error_values[(order,epoch)] = error
    
for ax in axs.flat:
    ax.set(xlabel='X', ylabel='Y')
plt.show()

#-- Plot Errors --
fig, ax = plt.subplots(1,1) 
x_e = list(range(1,len(list(error_values.values()))+1))
y_e = list(error_values.values())
ax.scatter(x = x_e, y = y_e)
ax.plot(x_e , y_e)
plt.title("MSE")
plt.xlabel("(Order,Epoch)")
plt.ylabel("MSE")
ax.set_xticks(list(range(1,len(list(error_values.values()))+1)))
x_ticksLabels = []
for k in error_values.keys():
    x_ticksLabels.append("("+str(k[0])+","+str(k[1])+")")
ax.set_xticklabels(x_ticksLabels, rotation='vertical')
plt.show()

#-- Test different Regularization factors ---- --------------------------------
order = 5
epoch = 10000
regular_values = [0.01,2,10]
colors =['red','black','blue']

fig = plt.figure(figsize=(5,5))
#-- Plot Main DS --
plt.scatter(x=X, y=y)


for i in range(len(regular_values)):
    reg= regular_values[i]
    
    #-- log --
    print("Regulazrization Factor = %f ------------------------" %(reg))
    
    #-- Run Gradient Descent and Get theta values --
    theta = Regression_GradientDescent(x=x_train, y=y_train, order=order, epoch=epoch,
                                       leraning_rate=0.01,regular_rate=reg)
    
    #-- Use theta values to predict y for all data --
    x_powers = X_Powers(X, order=order)
    y_pred = H(order=order,x=X,theta = theta,x_powers = x_powers)

    #-- Plot Results --        
    plt.scatter(x=X , y = y_pred,c=str(colors[i]),s=2,label = (str(reg)+'(' +str(colors[i])+')'))
    
    #-- Get MSE error for test data --
    error = Cost_GD(x_test, y_test, order=order,theta = theta)       
    print('\tMSE= ' + str(np.round(error,4)))
    
plt.legend(loc='upper left')
plt.show()


