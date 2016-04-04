# -*- coding: utf-8 -*-
import os

for subdir, dirs, files in os.walk('.'):
    if subdir == '.':
        print(files)