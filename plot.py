#! /usr/bin/env python
import pylab
import sys

f=open(sys.argv[1], 'r')
ky=[]
E=[]
f.readline()
for line in f.readlines():
	ky.append(float(line.split()[0]))
	E.append(float(line.split()[3]))
pylab.plot(ky, E, '.')
pylab.xlabel('ky')
pylab.ylabel('Energy')
pylab.title('Usual quantum Hall')
pylab.show()

f.close()
