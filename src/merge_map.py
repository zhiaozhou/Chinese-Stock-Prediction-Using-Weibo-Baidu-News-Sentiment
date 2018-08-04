#!/share/apps/anaconda3/4.3.1/bin/python

import pandas as pd
import datetime
import numpy as np
from collections import defaultdict

output = defaultdict(lambda: [0,0])
for i in sys.stdin:
    for i in f:
        if len(i.split('+')) >= 9:
            date = i.split('+')[-6]
            sentiment = int(i.split('+')[-5])
            ticker = i.split('+')[-1][:6]
            key = ticker+','+date
            output[key][0] += sentiment
            output[key][1] += 1
for i,j in dict(output).items():
    print(i,',',j)