#!/share/apps/anaconda3/4.3.1/bin/python
# -*- coding: utf-8 -*-

import sys
from datetime import datetime

for i in sys.stdin:
    x,y = i.split('+++++')
    x = x.strip()
    y = y.strip()
    if not ((len(x) == 0) | (len(y)<19)):
        y = datetime.strptime(y,"%Y-%m-%d %H:%M:%S")
        print('{0},{1}'.format(x,y))