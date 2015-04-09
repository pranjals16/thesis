#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

read_files1 = glob.glob("Jagran/pos/*")
read_files2 = glob.glob("Jagran/neg/*")
read_files3 = glob.glob("Navbharat/pos/*")
read_files4 = glob.glob("Navbharat/neg/*")

#with open("testData.tsv", "wb") as outfile:
#	outfile.write("sentiment\treview\n")
with open("new_movie_reviews.txt", "a") as outfile:
    for f in read_files1:
		with open(f, "rb") as infile:
			outfile.write(infile.read())
			
with open("new_movie_reviews.txt", "a") as outfile:
    for f in read_files2:
		with open(f, "rb") as infile:
			outfile.write(infile.read())
			
with open("new_movie_reviews.txt", "a") as outfile:
    for f in read_files3:
		with open(f, "rb") as infile:
			outfile.write(infile.read())
with open("new_movie_reviews.txt", "a") as outfile:
    for f in read_files4:
		with open(f, "rb") as infile:
			outfile.write(infile.read())
