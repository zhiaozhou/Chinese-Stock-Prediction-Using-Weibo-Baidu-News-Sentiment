import tushare as ts # pip install tushare
import pandas as pd
import numpy as np
import time

stk_list = ts.get_stock_basics().index # get_china_a_share_stock_list
writer = pd.ExcelWriter('./stock_price.xlsx', engine = 'xlsxwriter')
j = 0
for i in stk_list:
    price = ts.get_k_data(i,start='2015-01-01') # save the price dataframe of a specific stock into the excel file
    price.to_excel(writer, sheet_name = i)
    if j % 20 == 0:
        print('{}:{} finished - {}'.format(j,3533,i))
        b = time.time()
        print('{}s taken, about {}s for total'.format((b-a),3533*((b-a)/20)))
        a = time.time()
    j += 1
writer.save()