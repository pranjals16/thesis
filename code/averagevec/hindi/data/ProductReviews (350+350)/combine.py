#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

with open("trainData.tsv", "wb") as outfile:
	outfile.write("sentiment\treview\n")
with open("trainData.tsv", "a") as outfile:
    f=open("Positive.txt","r")
    for line in f:
		outfile.write("1\t")
		outfile.write(line)
with open("trainData.tsv", "a") as outfile:
    f=open("Negative.txt","r")
    for line in f:
		outfile.write("0\t")
		outfile.write(line)
