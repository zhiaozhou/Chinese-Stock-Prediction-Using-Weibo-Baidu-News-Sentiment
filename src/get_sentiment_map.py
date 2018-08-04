#!/usr/bin/env python 

for i in sys.stdin:
    data = (i.strip()[2:-2][:-23], i.strip()[2:-2][-19:])
    print('%s+%s' % (data[0].replace('+',''),data[1][:-9]))
        