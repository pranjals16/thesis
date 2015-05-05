#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob

read_files1 = glob.glob("Jagran/pos/*")
read_files2 = glob.glob("Jagran/neg/*")
read_files3 = glob.glob("Navbharat/pos/*")
read_files4 = glob.glob("Navbharat/neg/*")

with open("Positive.txt", "a") as outfile:
	for f in read_files1:
		sent=[]
		with open(f, "rb") as infile:
			for line in infile:
				sent.append(line.rstrip())
			for line in sent:
				outfile.write(line)
			outfile.write("\n")
with open("Negative.txt", "a") as outfile:
	for f in read_files2:
		sent=[]
		with open(f, "rb") as infile:
			for line in infile:
				sent.append(line.rstrip())
			for line in sent:
				outfile.write(line)
			outfile.write("\n")
with open("Positive.txt", "a") as outfile:
	for f in read_files3:
		sent=[]
		with open(f, "rb") as infile:
			for line in infile:
				sent.append(line.rstrip())
			for line in sent:
				outfile.write(line)
			outfile.write("\n")
with open("Negative.txt", "a") as outfile:
	for f in read_files4:
		sent=[]
		with open(f, "rb") as infile:
			for line in infile:
				sent.append(line.rstrip())
			for line in sent:
				outfile.write(line)
			outfile.write("\n")
