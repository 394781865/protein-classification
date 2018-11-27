# -*- coding: utf-8 -*-

import os

file = './log/1543146826/second_stage.txt'
f = open(file, 'r')
for line in f.readlines():
    line = line.strip()
    try:
        if int(line.split('/')[0])<1166:
            continue
    except:
        print(line)
        continue
    print(line)


