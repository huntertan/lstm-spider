#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import cookielib
import os
import random
import urllib2

import sqlite3
from bs4 import BeautifulSoup
import time
from selenium import webdriver


class CommunityPriceSpider:

    driver = None

    host = "http://hz.lianjia.com/"
    defaultFilename = 'xiaoquInfo.txt'
    basicUrl = "http://hz.lianjia.com/chengjiao/"

    filePath = "chargeInfo.txt"

    def generatorPhantomJSDriver(self, driverDir):
        os.environ["webdriver.phantomjs.driver"] = driverDir
        return webdriver.PhantomJS(driverDir)

    def generatorChromeDriver(self, driverDir):
        os.environ["webdriver.chrome.driver"] = driverDir
        return webdriver.Chrome(driverDir)

    def __init__(self):
        driverDir = "/Users/hanqing.thq/PycharmProjects/hoursePricePrediction/phantomjs"
        self.driver = self.generatorPhantomJSDriver(driverDir)

    def readCommunityIdFromFile(self):
        fp = codecs.open(self.defaultFilename, 'r', 'utf-8')
        xiaoquId = []
        for line in fp:
            infos = line.split(" ")
            xiaoquId.append(infos[0])
        return xiaoquId

    def requestCommunityPrice(self, xiaoquIds):
        print xiaoquIds
        print xiaoquIds.__len__()
        self.saveChargeInfos([],'w')
        # get each xiaoqu charge info.
        for id in xiaoquIds:
            sleepSecond = random.randint(10, 100)
            time.sleep(sleepSecond)
            print "sleep "+ str(sleepSecond)+" second"
            print "begin to grab charge info of xiaoqu:" + str(id)
            priceUrl = self.basicUrl + "c" + id + "/"
            print "access to url:" + priceUrl + " " + "xiaoquId:" + id
            self.driver.get(priceUrl)
            data = self.driver.page_source
            soup = BeautifulSoup(data, "html.parser")
            # all charge amount
            totalCount = soup.find('div', attrs={'class': 'total fl'}).span.contents[0]
            priceListContent = soup.find('ul', attrs={'class':'listContent'})
            priceList = priceListContent.findAll('li')
            # charge number of one page.
            pageSize = len(priceList)

            chargeList = []
            for priceItem in priceList :
                title = priceItem.find('div',attrs={'class':'title'})
                priceDetailUrl = title.a['href']
                tokens = str(priceDetailUrl).replace(".html","").split("/")
                size = len(tokens)
                if size < 1 :
                    return None;
                # unique id for charge
                chargeId = tokens[size-1]
                chargeHourse = title.a.contents[0]
                price = priceItem.find('div',attrs={'class':'totalPrice'}).span.contents[0]
                dealDate = priceItem.find('div',attrs={'class':'dealDate'}).contents[0]
                chargeList.append([id,chargeId,price,chargeHourse,dealDate])

            # save charge info at the end of the file
            self.saveChargeInfos(chargeList,'a+')

            # charge list more than one page
            if int(pageSize) < 1 :
                continue
            if int(totalCount) % int(pageSize) == 0 :
                pageNum = int(totalCount) / int(pageSize)
            else :
                pageNum = int(totalCount) / int(pageSize) + 1
            if pageNum > 1 :
                for i in range(2,pageNum) :
                    pgUrl = self.basicUrl + "pg" + str(i) + "c" + id + "/"
                    chargeSubList = self.getAndParseChargeInfo(pgUrl)
                    chargeList = chargeList + chargeSubList

        return chargeList

    def getAndParseChargeInfo(self,url):
        sleepSecond = random.randint(10, 100)
        time.sleep(sleepSecond)
        print "sleep " + str(sleepSecond) + " second"
        print "access to url:" + url
        # do the request
        self.driver.get(url)
        data = self.driver.page_source
        soup = BeautifulSoup(data, "html.parser")
        priceListContent = soup.find('ul', attrs={'class':'listContent'})
        priceList = priceListContent.findAll('li')

        chargeList = []
        for priceItem in priceList:
            title = priceItem.find('div', attrs={'class': 'title'})
            priceDetailUrl = title.a['href']
            tokens = str(priceDetailUrl).replace(".html", "").split("/")
            size = len(tokens)
            if size < 1:
                return None;
            # unique id for charge
            chargeId = tokens[size - 1]
            chargeHourse = title.a.contents[0]
            price = priceItem.find('div', attrs={'class': 'totalPrice'}).span.contents[0]
            dealDate = priceItem.find('div', attrs={'class': 'dealDate'}).contents[0]
            chargeList.append([id, chargeId, price, chargeHourse, dealDate])

        self.saveChargeInfos(chargeList,'a+')

        return chargeList

    def saveChargeInfos(self, chargeInfos, model):
        fp = codecs.open(self.filePath, model, 'utf-8')
        for info in chargeInfos:
            strLine = info[0].__str__() + " " + info[1] + " " + info[2] + " " + info[3] + " " + info[4] + '\n'
            fp.write(strLine)
        fp.close()

priceSpider = CommunityPriceSpider()
xiaoquIds = priceSpider.readCommunityIdFromFile()
priceSpider.requestCommunityPrice(xiaoquIds)