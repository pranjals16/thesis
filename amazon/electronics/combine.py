#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

with open("testData.tsv", "wb") as outfile:
	outfile.write("sentiment\treview\n")
with open("testData.tsv", "a") as outfile:
    f=open("test-pos.txt","r")
    for line in f:
		outfile.write("1\t")
		outfile.write(line)
with open("testData.tsv", "a") as outfile:
    f=open("test-neg.txt","r")
    for line in f:
		outfile.write("0\t")
		outfile.write(line)
