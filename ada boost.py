''' boosting version :1  date: 3/02 18:00'''
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
class AdaBoost():
    def __init__(self,M=3):
        self.M=M


    def fit(self,X,y):
        Model_list=[]
        weights = np.full(X.shape[0], (1 / X.shape[0]))
        self.alpha=[]
        for i in range(self.M):
            model = LinearSVC()
            model.fit(X,y)
            y_hat=model.predict(X)
            #error_vec=(y_hat!=y).astype('int')
            #weighted_error=np.dot(weights,error_vec)/np.sum(weights)#weights[i]*(1-sklearn.metrics.accuracy_score(y,model.predict(X)))
            weighted_error=np.sum((1 - sklearn.metrics.accuracy_score(y,y_hat))*weights)
            alpha=0.5*np.log((1-weighted_error)/weighted_error)
            self.alpha.append(alpha)
            vec=np.exp(-alpha * np.multiply(y, model.predict(X)))
            z = np.dot(vec,weights)
            weights = np.multiply(weights,vec/z)

            Model_list.append(model)
        self.Model_list=Model_list
        return


    def predict(self,x):
        pred=np.zeros((x.shape[0],len(self.Model_list)))
        for i,model in enumerate(self.Model_list):
            pred[:,i]=self.alpha[i]*model.predict(x)
            # stats.mode(np.sign(pred.sum(axis=1)).T)[0]
        return np.sign(pred.sum(axis=1))



if __name__ == "__main__":
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=1500, noise=0.2, random_state=101, factor=0.5)
    '''splitting data train and test'''
    y[y==0]=-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=10)

    LS = LinearSVC()
    LS.fit(X_train, y_train)
    y_pred = LS.predict(X_test)

    AD = AdaBoost(8)
    AD.fit(X_train, y_train)
    y_hat_test = AD.predict(X_test)
    print('our adaboost prediction ', sklearn.metrics.accuracy_score(y_test, y_hat_test.reshape(-1, 1)))
    print('sklearn LinearSVC prediction ', sklearn.metrics.accuracy_score(y_test, y_pred))
    print('bp')