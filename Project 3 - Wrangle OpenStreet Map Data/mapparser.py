#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case Study Code 
This  will find out the number of unique tags in the dataset. 
"""
import xml.etree.cElementTree as ET
import pprint
from collections import defaultdict

def count_tags(filename):
        # My Code
    countData = defaultdict(int)
    for event, node in ET.iterparse(filename):
        if event == 'end': 
            countData[node.tag]+=1
        node.clear()             
    return countData


pprint.pprint(count_tags('sample-new-delhi.osm'))
