import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import matplotlib.cm as cm
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import warnings
warnings.filterwarnings('ignore')

from memoization import cached

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN , OPTICS
from sklearn.manifold import TSNE 
from sklearn import preprocessing


def PCA_normalize(df_returns, n_components=50):
    df_ret_train_normalized = preprocessing.StandardScaler().fit_transform(df_returns)
    pca = PCA(n_components=n_components)
    pca.fit(df_ret_train_normalized)
    X = pca.components_.T    
    return X 

def normalize(df_returns):
    # normalize the different stock return of one day
    df_ret_train_normalized = preprocessing.StandardScaler().fit_transform(df_returns.T)
    return df_ret_train_normalized.T

def DBSCAN_fit(df_returns, eps=2, min_samples=2):
    X = df_returns.T
    clf = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clf.labels_
    return labels

def OPTICS_fit(df_returns, min_samples=4):
    X = df_returns.T
    clf = OPTICS(min_samples=4).fit(X)
    labels = clf.labels_
    return labels


def get_hedgeRatio(df):
    """
    get PCA hedge Ratio
    """
    pca = PCA().fit(df)
    hedgeRatio = pca.components_[0][1] / pca.components_[0][0]
    return hedgeRatio


def Cointegration(Price, cluster, significance, start_day, end_day): 
    pair_coin = []
    p_value = []
    n = cluster.shape[0] 
    keys = cluster.keys() 
    for i in range(n):
        for j in range(i+1,n):
            asset_1 = Price.loc[start_day:end_day, keys[i]] 
            asset_2 = Price.loc[start_day:end_day, keys[j]] 
            results = sm.OLS(asset_1, asset_2).fit() 
            predict = results.predict(asset_2)
            error = asset_1 - predict
            ADFtest = ts.adfuller(error)
            if ADFtest[1] < significance:
                pair_coin.append([keys[i], keys[j]])
                p_value.append(ADFtest[1]) 
    return p_value, pair_coin


def PairSelection(Price, ticker_count_reduced, clustered_series, significance, start_day, end_day, E_selection):
    Opt_pairs = [] # to get best pair in cluster i
    if E_selection == True: # select one pair from each cluster 
        for i in range(len(ticker_count_reduced)):
            cluster = clustered_series[clustered_series == i]
            keys = cluster.keys()
            result = Cointegration(Price, cluster, significance, start_day, end_day) 
            if len(result[0]) > 0:
                if np.min(result[0]) < significance:
                    index = np.where(result[0] == np.min(result[0]))[0][0] 
                    Opt_pairs.append([result[1][index][0], result[1][index][1]])
    else:
        p_value_contval = []
        pairs_contval = []
        for i in range(len(ticker_count_reduced)):
            cluster = clustered_series[clustered_series == i]
            keys = cluster.keys()
            result = Cointegration(Price, cluster, significance, start_day, end_day)

            if len(result[0]) > 0: 
                p_value_contval += result[0] 
                pairs_contval += result[1]
        Opt_pairs = [x for _, x in sorted(zip(p_value_contval, pairs_contval))]
    
    return Opt_pairs