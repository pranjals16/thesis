#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

read_files = glob.glob("neg/*")

#with open("testData.tsv", "wb") as outfile:
#	outfile.write("sentiment\treview\n")
with open("testData.tsv", "a") as outfile:
    for f in read_files:
		with open(f, "rb") as infile:
			outfile.write("0\t")
			#for line in infile:
			#	line=line.replace("\t"," ")
			#outfile.write(line)
			outfile.write(infile.read())
			outfile.write("\n")



