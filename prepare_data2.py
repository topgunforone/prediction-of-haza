#encoding:utf-8
import numpy as np
import pandas as pd
import random
random.seed(2018)

def shift(df ,num =4):
    df =df.values.reshape(-1,3,100) #transfor to one day
    len_df =df.shape[0]
    res  = []
    for  i in range(num*3-1,len_df):
        res.append(df[i-(num*3-1):(i+1),:])
    return np.array(res)

def prepare_data(shift_num =4):
    pm = pd.read_csv('getdata/sites4_getdata.csv',index_col=0,names=[1,2,3,4],header =0,dtype=np.float)
    t_csv = pd.read_csv('getdata/t_getdata.csv',index_col=[0,1],names=list(range(100)),header =0,dtype=np.float)
    v_csv = pd.read_csv('getdata/v_getdata.csv',index_col=[0,1],names=list(range(100)),header =0,dtype=np.float)
    z_csv = pd.read_csv('getdata/z_getdata.csv',index_col=[0,1],names=list(range(100)),header =0,dtype=np.float)
    #pm = pm.values.reshape(-1,1)
    t_csv.iloc[:9,:] = np.nan
    t_csv.iloc[-3:, :] = np.nan
    t_csv.dropna(inplace = True)
    v_csv.iloc[:9, :] = np.nan
    v_csv.iloc[-3:, :] = np.nan
    v_csv.dropna(inplace = True)
    z_csv.iloc[:9, :] = np.nan
    z_csv.iloc[-3:, :] = np.nan
    z_csv.dropna(inplace = True)
    pm.iloc[0,:] =np.nan
    pm.dropna(inplace = True)
    y = pm.values.reshape(-1,1)

    t_csv = shift(t_csv,shift_num)
    v_csv= shift(v_csv,shift_num)
    z_csv = shift(z_csv,shift_num)
    y = y[shift_num*3-1:]
    reshape_t_csv = t_csv.reshape(-1,shift_num*3,3,10,10)
    reshape_v_csv = v_csv.reshape(-1,shift_num*3,3,10,10)
    reshape_z_csv = z_csv.reshape(-1,shift_num*3,3,10,10)



    x = np.concatenate((reshape_t_csv,reshape_v_csv,reshape_z_csv),2)#[n,12,9,10,10]
    x = np.transpose(x,[0,1,3,4,2])  # [n,12,10,10,9]
    total_num = x.shape[0]
    index= list(range(total_num))
    random.shuffle(index)
    spilt =0.7

    train_end = int(total_num*spilt)
    train_end = (train_end//32)*32
    train_x = x[index[:train_end],:,:,:]
    train_y = y[index[:train_end]]
    val_end = (total_num-train_end)//32*32+train_end
    test_x = x[index[train_end:val_end]]
    test_y = y[index[train_end:val_end]]
    return  train_x,train_y,test_x,test_y
if __name__=='__main__':
    prepare_data()