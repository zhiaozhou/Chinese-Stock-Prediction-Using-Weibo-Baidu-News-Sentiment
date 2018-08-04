#!/usr/bin/env python 

import sys
from datetime import datetime 

types_col1 =[]
types_col2 = []
max_len_str = 0
time_range = [datetime(2018,5,1),datetime(2018,5,1)]
n_samples = 0
num_bad_record = 0
    
for i in sys.stdin:
    x,y = i.split('+++++')
    x = x.strip()
    y = y.strip()
    if ((len(x) == 0) | (len(y)<19)):
        num_bad_record += 1
    else:
        y = datetime.strptime(y,"%Y-%m-%d %H:%M:%S")
        type_i_col1 = type(x)
        type_i_col2 = type(y)
        len_i_col1 = len(x)
        if type_i_col1 not in types_col1:
            types_col1.append(type_i_col1)
        if type_i_col2 not in types_col2:
            types_col2.append(type_i_col2)
        if len_i_col1 > max_len_str:
            max_len_str = len_i_col1
        if y < time_range[0]:
            time_range[0] = y
        elif y > time_range[1]:
            time_range[1] = y
        n_samples += 1
print('the type of column 1 is %s' % (str(types_col1)))
print('the type of column 2 is %s' % (str(types_col2)))
print('the maximum length of the strings is %s' % (str(max_len_str)))
print('the time range in the dataset is %s' % (str(time_range)))
print('the number of samples is %s' % (str(n_samples)))
print('the number of bad records is %s' % (str(num_bad_record)))