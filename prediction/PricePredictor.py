#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class PricePredictor:
    # lstm param
    timeStep = 20
    hiddenUnitSize = 10 #隐藏层神经元数量
    batchSize = 60  #每一批次训练多少个样例
    inputSize = 1 #输入维度
    outputSize=1 #输出维度
    lr = 0.0006 #学习率
    train_x, train_y = [],[] #训练数据集
    sortedChargeList = []  #排序的训练数据集
    normalizeData = []    #归一化的数据
    # dataFile = '/Users/hanqing.thq/github/lstm-spider/chargeInfo.txt'
    dataFile = '../chargeInfo.txt'
    date2Price = {}       #日期－每平米的价格映射
    chargeList = []       #交易价格
    date2Charge = {}      #日期－交易价格映射
    meanPrice = 0        #均价
    stdPrice = 0
    X = tf.placeholder(tf.float32, [None, timeStep, inputSize])
    Y = tf.placeholder(tf.float32, [None, timeStep, inputSize])
    weights = {
        'in': tf.Variable(tf.random_normal([inputSize, hiddenUnitSize])),
        'out': tf.Variable(tf.random_normal([hiddenUnitSize, 1]))
    }

    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[hiddenUnitSize, ])),
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

    # 构造数据
    def buildTrainDataSet(self):
        data = []
        for price in self.sortedChargeList:
            data.append(price[3])

        self.meanPrice = np.mean(data);
        self.stdPrice = np.std(data)
        self.normalizeData = (data - self.meanPrice) / self.stdPrice #标准化

        self.normalizeData = self.normalizeData[:,np.newaxis]  #增加维度
        for i in range(len(self.normalizeData)-self.timeStep-1):
            x=self.normalizeData[i:i+self.timeStep]
            y=self.normalizeData[i+1:i+self.timeStep+1]
            self.train_x.append(x.tolist())
            self.train_y.append(y.tolist())

    # lstm算法定义
    def lstm(self, batchSize = None):
        if batchSize is None :
            batchSize = self.batchSize
        weightIn = self.weights['in']
        biasesIn = self.biases['in']
        input = tf.reshape(self.X, [-1,self.inputSize])
        inputRnn=tf.matmul(input,weightIn)+biasesIn
        inputRnn=tf.reshape(inputRnn,[-1,self.timeStep,self.hiddenUnitSize])  #将tensor转成3维，作为lstm cell的输入
        cell=tf.nn.rnn_cell.BasicLSTMCell(self.hiddenUnitSize)
        initState=cell.zero_state(batchSize,dtype=tf.float32)
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, inputRnn,initial_state=initState, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output=tf.reshape(output_rnn,[-1,self.hiddenUnitSize]) #作为输出层的输入
        w_out=self.weights['out']
        b_out=self.biases['out']
        pred=tf.matmul(output,w_out)+b_out
        return pred,final_states

    # 训练模型
    def trainLstm(self) :
        pred,_ = self.lstm()
        #定义损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.Y, [-1])))
        #定义训练模型
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 重复训练100次，训练是一个耗时的过程
            for i in range(100):
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

    def prediction(self):
        pred, _ = self.lstm(1)  # 预测时只输入[1,time_step,inputSize]的测试数据
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint('/Users/hanqing.thq/github/lstm-spider/prediction/')
            saver.restore(sess, module_file)
            # 取训练集最后一行为测试样本. shape=[1,time_step,inputSize]
            prev_seq = self.train_x[-1]
            predict = []
            # 得到之后100个预测结果
            for i in range(100):
                next_seq = sess.run(pred, feed_dict={self.X: [prev_seq]})
                predict.append(next_seq[-1])
                # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
            # 以折线图表示结果
            plt.figure()
            true_price = self.stdPrice**predict
            true_price = [price + self.meanPrice for price in true_price]
            plt.plot(list(range(len(self.normalizeData), len(self.normalizeData) + len(predict))), true_price, color='r')
            plt.show()

predictor = PricePredictor()
predictor.loadData()
# sortedChargeList = sorted(predictor.chargeList, key=predictor.getKey, reverse=False)
# 构建训练数据
predictor.buildTrainDataSet()
# 模型训练
# predictor.trainLstm()

# 预测－预测前需要先完成模型训练
predictor.prediction()



