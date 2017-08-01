#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class PricePredictor:
    # lstm param
    timeStep = 20
    rnnUnit = 10 #hidden layer units
    batchSize = 60  #每一批次训练多少个样例
    input_size = 1 #输入维度
    output_size=1 #输出维度
    lr = 0.0006 #学习率
    train_x, train_y = [],[] #训练数据集
    sortedChargeList = []

    dataFile = '/Users/hanqing.thq/github/lstm-spider/chargeInfo.txt'
    date2Price = {}
    chargeList = []
    date2Charge = {}

    X = tf.placeholder(tf.float32, [None, timeStep, input_size])
    Y = tf.placeholder(tf.float32, [None, timeStep, input_size])
    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnnUnit])),
        'out': tf.Variable(tf.random_normal([rnnUnit, 1]))
    }

    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnnUnit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    def loadData(self):
        fp = codecs.open(self.dataFile, 'r', 'utf-8')
        line = fp.readline()

        # parse line to data
        while line:
            line = fp.readline()
            print line
            data = line.split(" ")
            if len(data) < 7:
                continue
            area = float(data[5].encode('utf-8').replace("平米", ""))
            price = float(data[2])
            pricePerSquare = price / area
            charge = [str(data[1]), data[6].encode('utf-8').replace('\n', ''), data[3].encode('utf-8'), pricePerSquare]
            self.chargeList.append(charge)
            self.date2Charge[str(data[1])] = charge  # date: {name:price}
            self.date2Price[str(data[1])] = pricePerSquare

        self.sortedChargeList = sorted(self.chargeList, key=predictor.getKey, reverse=False)

    def getKey(self, item):
        return item[1]

    def buildTrainDataSet(self):
        data = []
        for price in self.sortedChargeList:
            data.append(price[3])

        normalize_data = (data - np.mean(data)) / np.std(data) #标准化

        plt.figure()
        plt.plot(normalize_data)
        plt.show()

        normalize_data = normalize_data[:,np.newaxis]  #增加维度
        for i in range(len(normalize_data)-self.timeStep-1):
            x=normalize_data[i:i+self.timeStep]
            y=normalize_data[i+1:i+self.timeStep+1]
            self.train_x.append(x.tolist())
            self.train_y.append(y.tolist())

    def lstm(self):
        weightIn = self.weights['in']
        biasesIn = self.biases['in']
        input = tf.reshape(self.X, [-1,self.input_size])
        input_rnn=tf.matmul(input,weightIn)+biasesIn
        input_rnn=tf.reshape(input_rnn,[-1,self.timeStep,self.rnnUnit])  #将tensor转成3维，作为lstm cell的输入
        cell=tf.nn.rnn_cell.BasicLSTMCell(self.rnnUnit)
        init_state=cell.zero_state(self.batchSize,dtype=tf.float32)
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output=tf.reshape(output_rnn,[-1,self.rnnUnit]) #作为输出层的输入
        w_out=self.weights['out']
        b_out=self.biases['out']
        pred=tf.matmul(output,w_out)+b_out
        return pred,final_states

    def trainLstm(self) :
        pred,_ = self.lstm()
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 重复训练10000次
            for i in range(10000):
                step = 0
                start = 0
                end = start + self.batchSize
                while end < len(self.train_x):
                    _, loss_ = sess.run([train_op, loss], feed_dict={self.X: self.train_x[start:end], self.Y: self.train_y[start:end]})
                    start += self.batchSize
                    end = start + self.batchSize
                    # 每10步保存一次参数
                    if step % 10 == 0:
                        print(i, step, loss_)
                        print("保存模型：", saver.save(sess, 'stock.model'))
                    step += 1

predictor = PricePredictor()
predictor.loadData()

# sortedChargeList = sorted(predictor.chargeList, key=predictor.getKey, reverse=False)

predictor.buildTrainDataSet()



