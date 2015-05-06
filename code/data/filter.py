#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
f=open('hindi_iitb','r')
for line in f:
	line=line.replace('.','\n')
	line=line.replace('?','\n')
	line=line.replace('।','\n')
	print line
