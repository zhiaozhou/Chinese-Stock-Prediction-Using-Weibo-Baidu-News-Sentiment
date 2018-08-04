#!/share/apps/anaconda3/4.3.1/bin/python

import sys
import ast

mid_set = []
for line in sys.stdin:
    line = ast.literal_eval(line.strip())
    if line['mid'] not in mid_set:
        mid_set.append(line['mid'])
        print(str(line))
    
    