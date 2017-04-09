#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import random
from bs4 import BeautifulSoup
import time
from selenium import webdriver


class CommunitySpider:
    cookie = None;
    openner = None;
    host = "http://hz.lianjia.com/"
    driver = None

    def generatorPhantomJSDriver(self, driverDir):
        os.environ["webdriver.phantomjs.driver"] = driverDir
        return webdriver.PhantomJS(driverDir)

    def generatorChromeDriver(self, driverDir):
        os.environ["webdriver.chrome.driver"] = driverDir
        return webdriver.Chrome(driverDir)

    def __init__(self):
        driverDir = "/Users/hanqing.thq/PycharmProjects/hoursePricePrediction/phantomjs"
        self.driver = self.generatorPhantomJSDriver(driverDir)

        # http: // hz.lianjia.com / api / sidebarxiaoqu?cityId = 330100 & id = 330108 & type = district
    def requestComunityInfo(self):
        xiaoquList = []
        url = "http://hz.lianjia.com/xiaoqu/binjiang/"
        self.driver.get(url)
        data = self.driver.page_source
        soup = BeautifulSoup(data, "html.parser")
        totalCount = soup.find('h2', attrs={'class': 'total fl'}).span.contents[0]
        xiaoquTag = soup.find_all("li", attrs={'class': 'clear xiaoquListItem'})
        pageSize = len(xiaoquTag)
        for tag in xiaoquTag:
            titleSet = tag.findAll("div", attrs={'class': 'title'})
            for title in titleSet:
                xiaoquUrl = title.a['href']
                xiaoquId = str(xiaoquUrl).split('/')[4]
                xiaoquName = title.a.contents[0]
                xiaoquList.append([xiaoquId,xiaoquName,xiaoquUrl])
        pageNum = int(totalCount) / pageSize + 1
        for i in range(2, pageNum):
            url = url + "pg" + str(i)
            xiaoquList = xiaoquList + self.getXiaoquInfo(url)
        self.driver.close()
        return xiaoquList

    def getXiaoquInfo(self, url):
        time.sleep(random.randint(10, 100))
        xiaoquList = []
        self.driver.get(url)
        data = self.driver.page_source
        soup = BeautifulSoup(data, "html.parser")
        xiaoquTag = soup.find_all("li", attrs={'class': 'clear xiaoquListItem'})
        for tag in xiaoquTag:
            titleSet = tag.findAll("div", attrs={'class': 'title'})
            for title in titleSet:
                xiaoquUrl = title.a['href']
                xiaoquId = str(xiaoquUrl).split('/')[4]
                xiaoquName = title.a.contents[0]
                xiaoquList.append([xiaoquId, xiaoquName, xiaoquUrl])
        self.driver.close()
        return xiaoquList

    def saveXiaoquInfoInFile(self, filePath, xiaoquInfoList):
        fp = codecs.open(filePath,'w','utf-8')
        print xiaoquInfoList.__len__()
        for info in xiaoquInfoList :
            strLine = info[0].__str__() + " " + info[1] + " " + info[2] + '\n'
            fp.write(strLine)
        fp.close()

spider = CommunitySpider()
allXiaoquInfo = spider.requestComunityInfo()
filepath = "xiaoquInfo.txt"
spider.saveXiaoquInfoInFile(filepath,allXiaoquInfo)