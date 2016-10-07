import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import math

sess = tf.InteractiveSession()

n_input = 30
n_step = 1
batch_size = 40

class Stock(object):
	
	def __init__(self, tr, expect, size):
		# set input data and size
		self.tr = tr
		self.expect = expect
		self.size = size
		
		# define input weights and bias
		#self.iw = tf.get_variable("input_w", [n_input, self.size])
		self.iw = tf.Variable(\
			tf.truncated_normal([n_input, self.size],\
			stddev=1.0 / math.sqrt(float(self.size))), name='iw')
		self.ib = tf.Variable(\
			tf.truncated_normal([self.size],\
			stddev=1.0 / math.sqrt(float(self.size))), name='ib')
		#self.ib = tf.get_variable("input_b", [self.size])

		# define output weights and bias
		self.ow = tf.Variable(\
			tf.truncated_normal([self.size, 1],\
			stddev=1.0 / math.sqrt(float(self.size))), name='ow')
		self.ob = tf.Variable(\
			tf.truncated_normal([1],\
			stddev=1.0 / math.sqrt(float(self.size))), name='ob')
		#self.ow = tf.get_variable("output_w", [self.size, 1])
		#self.ob = tf.get_variable("output_b", [1])


	def Lstm_training(self):
		# define placeholder
		data_in = tf.placeholder(tf.float32, shape=[None, n_step, \
			n_input], name='indat')
		data_out = tf.placeholder(tf.float32, shape=[None], name='outdat')
		istate = tf.placeholder("float", [None, 2*self.size])
			
		# calculate input for hidden layer
		data_in = tf.reshape(data_in, [-1, n_input])
		inputs = tf.matmul(data_in, self.iw) + self.ib

		# lstm cell	
		lstm_cell = rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0) 
		
		# split input of hidden layer because rnn cell needs a list of inputs
		inputs = tf.split(0, n_step, inputs)
		
		# get lstm cell output
		outputs, states = rnn.rnn(lstm_cell, inputs, initial_state=istate)

		# output layer
		data_o = tf.matmul(outputs[-1], self.ow) + self.ob
		#data_o = tf.transpose(data_o)
		
		# loss function
		loss = tf.reduce_mean(tf.square(data_o - tf.reshape(data_out, [-1])))
		regularizer = tf.nn.l2_loss(self.iw) + tf.nn.l2_loss(self.ib)\
			+ tf.nn.l2_loss(self.ow) + tf.nn.l2_loss(self.ob)
		loss += 2e-3 * regularizer
		# optimization
		optimizer = tf.train.GradientDescentOptimizer(2e-3).minimize(loss)

		sess.run(tf.initialize_all_variables())
		for i in range(5000):
			print "number of iteration: %d"%i
			offset = i%60	
			batch_x = self.tr[batch_size*offset:batch_size*(offset+1),:]
			batch_y = self.expect[batch_size*offset+30:batch_size*(offset+1)+30]
			optimizer.run(feed_dict={data_in: batch_x, data_out: batch_y, istate: np.zeros((batch_size, 2*self.size))})
			if i%10 == 0:
				loss_pr = loss.eval(feed_dict={\
					data_in: batch_x, data_out: batch_y, istate:\
					np.zeros((batch_size, 2*self.size))})
				print "step %d, training accuracy %g"%(i, loss_pr)
