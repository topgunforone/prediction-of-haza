#encoding:utf-8
import numpy as np
import pandas as pd
import random
random.seed(2018)
def prepare_data():
    pm = pd.read_csv('getdata/sites4_getdata.csv',index_col=0,names=[1,2,3,4],header =0,dtype=np.float)
    t_csv = pd.read_csv('getdata/t_getdata.csv',index_col=[0,1],names=list(range(100)),header =0,dtype=np.float)
    v_csv = pd.read_csv('getdata/v_getdata.csv',index_col=[0,1],names=list(range(100)),header =0,dtype=np.float)
    z_csv = pd.read_csv('getdata/z_getdata.csv',index_col=[0,1],names=list(range(100)),header =0,dtype=np.float)
    y = pm.values.reshape(-1,1)
    reshape_t_csv = t_csv.values.reshape(-1,10,10,3)
    reshape_v_csv = v_csv.values.reshape(-1,10,10,3)
    reshape_z_csv = z_csv.values.reshape(-1,10,10,3)
    x = np.concatenate((reshape_t_csv,reshape_v_csv,reshape_z_csv),axis =-1)
    total_num = x.shape[0]
    index= list(range(total_num))
    random.shuffle(index)
    spilt =0.7

    train_end = int(total_num*spilt)
    train_x = x[index[:train_end],:,:,:]
    train_y = y[index[:train_end]]

    test_x = x[index[train_end:]]
    test_y = y[index[train_end:]]
    return  train_x,train_y,test_x,test_y