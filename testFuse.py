# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 08:34:33 2018

@author: wangjinkui
"""
"""
地理导航坐标系为东北天(ENU)坐标系，投影到EN平面
State: 0:Ve  东向速度   单位m/s
       1:Vn  北向速度   单位m/s
       2:Pe  东向位置   单位m
       3:Pn  北向位置   单位m
       4:ax  载体的X方向加速度   单位 m/s^2
       5:ay  载体的Y方向加速度   单位 m/s^2
       6:wz  载体旋转角速度，旋转轴指向上，逆时针旋转为正   单位 deg/s
       7:fai 载体X轴与东向夹角，范围（-180°~180°）,由东向轴逆时针旋转到载体轴为正  单位deg/s
       8:time 时间，步长1ms
SensorNominal:
       0：ax 名义上X方向加速度 单位 m/s^2
       1：ay 名义上Y方向加速度 单位 m/s^2
       2：wz 名义上角速度      单位 deg/s
SensorNoisy:
       0：ax 加入噪声的X方向加速度 单位 m/s^2
       1：ay 加入噪声的Y方向加速度 单位 m/s^2
       2：wz 加入噪声的角速度      单位 deg/s
"""


deg2rad = 0.01745   #角度转弧度
rad3deg = 57.29578  #弧度转角度
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def CalOneStep(_StateOld, _SensorNominal, _dt):
    Phi = _StateOld[7] + _SensorNominal[2] * _dt
    ae  = _SensorNominal[0]*np.cos(Phi*deg2rad) - _SensorNominal[1]*np.sin(Phi*deg2rad)
    an  = _SensorNominal[0]*np.sin(Phi*deg2rad) + _SensorNominal[1]*np.cos(Phi*deg2rad)
    Ve  = _StateOld[0] + ae*_dt
    Vn  = _StateOld[1] + an*_dt
    Pe  = _StateOld[2] + 0.5*(_StateOld[0] + Ve)*_dt
    Pn  = _StateOld[3] + 0.5*(_StateOld[1] + Vn)*_dt
    time = _StateOld[8] +_dt
    StateNext = np.zeros(11)
    _SensorNoisy  = np.zeros(3)
    StateNext[0] = Ve
    StateNext[1] = Vn
    StateNext[2] = Pe
    StateNext[3] = Pn
    StateNext[4] = _SensorNominal[0]
    StateNext[5] = _SensorNominal[1]
    StateNext[6] = _SensorNominal[2]
    StateNext[7] = Phi
    StateNext[8] = time
    StateNext[9] = ae
    StateNext[10] = an
    _SensorNoisy[0] = _SensorNominal[0] +random.randint(-50,50)/100.0
    _SensorNoisy[1] = _SensorNominal[1] +random.randint(-50,50)/100.0
    _SensorNoisy[2] = _SensorNominal[2] +random.randint(-10,10)/100.0
    return StateNext,_SensorNoisy

SensorNominal_train = np.zeros(3)
SensorNoisyOneStep_train = np.zeros(3)
SensorLog_train = np.zeros((1,3))
StateOneStep_train = np.zeros(11)
StateLog_train = np.zeros((1,11))
dt = 0.001
time = 0
for i in range(20000):
    time = time+dt
    SensorNominal_train[0] = np.sin(time*5.0)*5
    SensorNominal_train[1] = np.cos(time*5.0)*10
    SensorNominal_train[2] = np.sin(time*5)*5+np.cos(time*10)*3
    StateOneStep_train,SensorNoisyOneStep_train = CalOneStep(StateOneStep_train,SensorNominal_train,dt)
    StateLog_train = np.row_stack((StateLog_train,StateOneStep_train))
    SensorLog_train = np.row_stack((SensorLog_train,SensorNoisyOneStep_train))

train_Xdata = SensorLog_train[0:10000,:]
train_Ydata = StateLog_train[0:10000,2]
test_Xdata  = SensorLog_train[10000:20000,:]
test_Ydata  = StateLog_train[10000:20000,2]
class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """
    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = 200        
        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 50

        # LSTM structure
        self.n_inputs = 3  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 128  # nb of neurons inside the neural network
        self.n_outputs = 1  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_outputs]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_outputs]))
        }


def LSTM_Network(tf_X, config):
    tf_X = tf.reshape(tf_X, [-1, config.n_inputs])
    tf_X = tf.matmul(tf_X, config.W['hidden']) + config.biases['hidden']
    tf_X = tf.reshape(tf_X,[-1,config.n_steps,config.n_hidden])
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    init_s = lstm_cell.zero_state(config.batch_size, dtype=tf.float32)    # very first hidden state 
    outputs, states = tf.nn.dynamic_rnn(lstm_cell,tf_X,initial_state = init_s,dtype=tf.float32)
    
    outputs = tf.reshape(outputs,[-1,config.n_hidden])
    outputs = tf.matmul(outputs, config.W['output']) + config.biases['output']
    # Linear activation
    return outputs

config = Config(train_Xdata, test_Xdata)
X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
Y = tf.placeholder(tf.float32, [None, config.n_steps, config.n_outputs])
train_Xdata = np.reshape(train_Xdata,[-1,config.n_steps,config.n_inputs])
train_Ydata = np.reshape(train_Ydata,[-1,config.n_steps,config.n_outputs])
pred_Y = LSTM_Network(X, config)
###
loss=tf.reduce_mean(tf.square(tf.reshape(pred_Y,[-1])-tf.reshape(Y, [-1])))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())     # initialize var in graph
    # Start training for each batch and loop epochs
for i in range(300):
    for index in range( config.n_steps-1):
        _,loss_out = sess.run([optimizer,loss], feed_dict={
                X:train_Xdata[index*config.batch_size:(index+1)*config.batch_size],
                Y:train_Ydata[index*config.batch_size:(index+1)*config.batch_size]
                })
        # Test completely at every epoch: calculate accuracy
 #   pred_out, loss_out = sess.run([pred_Y, loss],feed_dict={X: train_Xdata,Y: train_Ydata})
    print("traing iter: {},".format(i) +  " loss : {}".format(loss_out))
