#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

read_files = glob.glob("neg/*")

#with open("testData.tsv", "wb") as outfile:
#	outfile.write("sentiment\treview\n")
with open("hindi_review_movie2.txt", "a") as outfile:
    for f in read_files:
		with open(f, "rb") as infile:
			outfile.write(infile.read())
			#outfile.write("\n")



