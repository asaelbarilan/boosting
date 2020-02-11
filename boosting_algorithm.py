''' boosting version :3  date: 5/02 18:00'''
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

def l2boost(X,y,M=100,miu=0.1):
    model = DecisionTreeRegressor(max_depth=1)
    F_0=np.mean(y)
    FM_1=np.ones(len(y))*F_0
    for m in range(M):
        r=np.sum([y,-FM_1.reshape(-1,1)],axis=0)#negative gradient
        model.fit(X,r)
        F_M=np.sum([FM_1,miu*model.predict(X)],axis=0)
        FM_1=F_M
    return F_M


class Gboost():
    def __init__(self,n_trees=100,miu=0.1):
        self.n_trees=n_trees
        self.miu=miu


    def fit(self,X,y):
        model = DecisionTreeRegressor(max_depth=1)
        self.X=X
        self.y=y
        Model_list=[]
        # model.fit(X, y)
        # Model_list.append(model)

        F_0 = np.mean(y)
        self.mean_y=F_0
        FM_1 = np.ones(len(y)) * F_0  # or model.predict(X)
        #FM_1=model.predict(X)
        for m in range(self.n_trees):
            model = DecisionTreeRegressor(max_depth=1)
            r=np.sum([y,-FM_1.reshape(-1,1)],axis=0)#negative gradient
            model.fit(X,r)
            Model_list.append(model)
            F_M=np.sum([FM_1,self.miu*model.predict(X)],axis=0)
            FM_1=F_M
        self.Model_list=Model_list
        return


    def predict(self,x):
        pred=np.zeros((x.shape[0],len(self.Model_list)+1))
        pred[:, -1]=np.ones(x.shape[0])*self.mean_y
        for i,model in enumerate(self.Model_list):
            #y_pred=np.sum([y_pred,model.predict(x)],axis=0)
            pred[:,i]=self.miu*model.predict(x)
        return pred.sum(axis=1)




if __name__ == "__main__":
    data = load_boston
    X, y = load_boston(return_X_y=True)
    y = y.reshape(-1, 1)

    '''splitting data train and test'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


    #
    # #1st part-algorithm impelentation with comparison to tree
    # DT = DecisionTreeRegressor(max_depth=1)
    # DT.fit(X_train, y_train)
    # y_pred = DT.predict(X_test)
    # y_hat_l2boost = l2boost(X_train, y_train, M=100, miu=1)
    #
    # Gb = Gboost(100, 0.6)
    # Gb.fit(X_train, y_train)
    # y_hat_test = Gb.predict(X_test)
    # print('our l2boost train prediction', sklearn.metrics.mean_squared_error(y_train, y_hat_l2boost.reshape(-1, 1)))
    # print('sklearn DT train prediction ', sklearn.metrics.mean_squared_error(y_train, DT.predict(X_train)))
    #
    # print('our test prediction', sklearn.metrics.mean_squared_error(y_test, y_hat_test.reshape(-1, 1)))
    # print('sklearn DT test prediction ', sklearn.metrics.mean_squared_error(y_test, y_pred))

    # 2nd part-MSE as a function of the number of trees for a
    # logspace of n_trees up to 1,000. What is the optimal value of n_trees? of learning rate?

    score=pd.DataFrame(columns=['mse','lr','n_trees'])

    learning_rates=np.arange(0.1,1,0.1)
    trees=np.arange(1,1101,100)
    i=0
    for n_trees in trees:
        for lr in learning_rates:
            Gb = Gboost(n_trees, lr)
            Gb.fit(X_train, y_train)
            y_hat_test = Gb.predict(X_test)
            score.loc[i, 'n_trees']=n_trees
            score.loc[i,'lr']=lr
            score.loc[i,'mse']=(sklearn.metrics.mean_squared_error(y_test, y_hat_test.reshape(-1, 1)))
            i+=1
    score=score.sort_values(by='mse',ascending=True)
    print('best paramas')
    print(score.iloc[0, :])

    Mses=[]
    for n_trees in trees:
        Gb = Gboost(n_trees,0.1)
        Gb.fit(X_train, y_train)
        y_hat_test = Gb.predict(X_test)
        Mses.append(sklearn.metrics.mean_squared_error(y_test, y_hat_test.reshape(-1, 1)))

    plt.plot(np.log(trees),Mses)
    plt.title('MSE per iteration. learning rate:{}'.format(0.1))
    plt.xlabel('number of trees(log)')
    plt.ylabel('MSE')
    plt.show()
    print('bp')

    DT_Mses=[]
    depths=np.arange(1,20)
    for depth in depths:
        DT = DecisionTreeRegressor(max_depth=depth)
        DT.fit(X_train, y_train)
        y_pred = DT.predict(X_test)
        DT_Mses.append(sklearn.metrics.mean_squared_error(y_test,y_pred.reshape(-1, 1)))
    print('mse min value  of DecisionTreeRegressor for MaxDpeth {} is :'.format(depths[DT_Mses.index(min(DT_Mses))]),min(DT_Mses))

    print('DecisionTreeRegressor with MaxDpeth of 10 is better than our boosting algorithm' )