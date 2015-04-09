#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

with open("trainData_electronics.tsv", "wb") as outfile:
	outfile.write("sentiment\treview\n")
with open("trainData_electronics.tsv", "a") as outfile:
    f=open("pos.txt","r")
    for line in f:
		outfile.write("1\t")
		outfile.write(line)
with open("trainData_electronics.tsv", "a") as outfile:
    f=open("neg.txt","r")
    for line in f:
		outfile.write("0\t")
		outfile.write(line)
