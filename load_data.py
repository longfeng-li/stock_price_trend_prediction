from __future__ import division
import os
from sys import argv
from rnn import Stock
import time
import numpy as np
script, dataset = argv

def EMA(day, a):
	#print "length:%d"%length
	if len(a) == 1:
		return float(a[0])
	else:
		return 2/(day+1) * float(a[-1]) + (1-(2/(day+1)))*EMA(day, a[:-1])

print time.strftime('%H-%M-%S')
input = open(dataset, 'r')
line = input.readline()
stock_close = []
stock_single = []
i = 0
while line != '':
	stock_single.append(line.split('\t')[3])	
	i = i + 1 
	line = input.readline()
stock_single.reverse()

for i in range(len(stock_single)):
	if i >= 29:
		stock_close.append(stock_single[i-29:i+1])
#print stock_close
data_in = [[0]*5]*len(stock_close)
for i in range(len(stock_close)):
	#print stock_close[i]
	#print stock_close[i][-1]
	data_in[i][0] = EMA(26, stock_close[i])	
	print "EMA26:%f"%data_in[i][0]	
	data_in[i][1] = EMA(18, stock_close[i])		
	print "EMA18:%f"%data_in[i][1]	
	data_in[i][2] = EMA(12, stock_close[i])		
	print "EMA12:%f"%data_in[i][2]	
	data_in[i][3] = EMA(5, stock_close[i])	
	print "EMA5:%f"%data_in[i][3]	
	data_in[i][4] = float(stock_close[i][-1])
	#print "stock[-1]:%f"%float(stock_close[i][-1])
	#print data_in[i][4]
	
print "*****************"
#print data_in
print "*****************"	
stock_x = np.array(data_in)
stock_y = np.array(stock_single)
exe = Stock(stock_x, stock_y, 30)
exe.Lstm_training() 
print len(stock_close)
print time.strftime('%H-%M-%S')
