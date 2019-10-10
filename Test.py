#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:08:49 2019

@author: aryan
"""
import os
path='/home/aryan/Desktop/Stars'
# Workbook() takes one, non-optional, argument  
# which is the filename that we want to create. 
import openpyxl
wb = openpyxl.load_workbook(os.path.join(path , 'Error.xlsx'))
sheet = wb.active 

di=0
for i in range(25):
    x2=(cen[i][0]-c[i][0])**2
    y2=(cen[i][1]-c[i][1])**2
    di=di+(x2+y2)**0.5
avg=di/25

di=0
for i in range(25):
    x2=(cen[i][0]-c[i][0])**2
    y2=(cen[i][1]-c[i][1])**2
    di=di+(((x2+y2)**0.5)-avg)**2
var=di/25
print(avg,var)

e='A'+str(num)
v='B'+str(num)
sheet[e].value=avg
sheet[v].value=var

wb.save(os.path.join(path , 'Error.xlsx'))
num=num+1