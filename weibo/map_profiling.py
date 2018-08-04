#!/usr/bin/env python 

import sys

for i in sys.stdin:
    x = i.strip()[2:-2][:-23]
    y = i.strip()[2:-2][-19:]
    print(x+'+++++'+y)