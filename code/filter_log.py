# -*- coding: utf-8 -*-

import os

file = './log/1542798002/second_stage.txt'
f = open(file, 'r')
for line in f.readlines():
    line = line.strip()
    try:
        if int(line.split('/')[0])<1748:
            continue
    except:
        print(line)
        continue
    print(line)


