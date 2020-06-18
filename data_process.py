# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:00:55 2019

@author: huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


df = pd.read_csv("D:/UCI/235/data/lv.csv",sep=';') 
df.head


def yesNo(x):
    if x=="YES":
        return 1
    else:
        return 0
    
def toOrd(str):
    x=0
    for l in str:
        x += ord(l)
    return int(x)

cols = ['User country', 'Nr. reviews','Nr. hotel reviews','Helpful votes',
        'Score','Period of stay','Traveler type','Pool','Gym','Tennis court',
        'Spa','Casino','Free internet','Hotel name','Hotel stars','Nr. rooms',
        'User continent','Member years','Review month','Review weekday']

df['Casino']=df['Casino'].apply(lambda x : yesNo(x))
df['Gym']=df['Gym'].apply(lambda x : yesNo(x))
df['Pool']=df['Pool'].apply(lambda x : yesNo(x))
df['Tennis court']=df['Tennis court'].apply(lambda x : yesNo(x))
df['Casino']=df['Casino'].apply(lambda x : yesNo(x))
df['Free internet']=df['Free internet'].apply(lambda x : yesNo(x))
df['Spa']=df['Spa'].apply(lambda x : yesNo(x))


cols2 = ['Period of stay', 'Hotel name', 'User country',
         'Traveler type', 'User continent', 'Review month', 'Review weekday,']

for y in cols2:
    df[y]=df[y].apply(lambda x: toOrd(x))



df['sport']=0 
index1=np.where((df['Pool']=='YES') & (df['Gym']=='YES') & (df['Tennis court']=='YES'))[0]
df['sport'][index1]=1


if df['Pool']=='YES':
    df['sport']=1
df['sport'][trip$Pool=="YES"] & trip$Gym=="YES"& trip$Tennis.court=="YES"]   
    
df.to_csv('D:/UCI/235/data/tripAdvisorFL.csv')