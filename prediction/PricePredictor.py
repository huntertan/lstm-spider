#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import matplotlib.pyplot as plt
import numpy as np

class PricePredictor:
    # lstm param
    timeStep = 20
    rnnUnit = 10 #hidden layer units
    batchSize = 60  #每一批次训练多少个样例
    input_size = 1 #输入维度
    output_size=1 #输出维度
    lr = 0.0006 #学习率
    train_x, train_y = [],[] #训练数据集

    dataFile = '/Users/hanqing.thq/github/lstm-spider/chargeInfo.txt'
    date2Price = {}
    chargeList = []
    date2Charge = {}

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

    def getKey(self, item):
        return item[1]

    def buildTrainDataSet(self):
        data = []
        for price in sortedChargeList:
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

predictor = PricePredictor()
predictor.loadData()
sortedChargeList = sorted(predictor.chargeList, key=predictor.getKey, reverse=False)

predictor.buildTrainDataSet()



