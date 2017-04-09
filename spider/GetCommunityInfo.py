#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import cookielib
import random
import urllib2
from bs4 import BeautifulSoup
import time


class CommunitySpider:
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Host': 'hz.lianjia.com'
    };
    cookie = None;
    openner = None;
    host = "http://hz.lianjia.com/";

    def __init__(self):
        cookieFileName = 'cookie.txt'
        self.cookie = cookielib.MozillaCookieJar(cookieFileName)
        handler = urllib2.HTTPCookieProcessor(self.cookie)
        self.opener = urllib2.build_opener(handler)
        self.opener.open(self.host)
        self.cookie.save(ignore_discard=True, ignore_expires=True)

    # http: // hz.lianjia.com / api / sidebarxiaoqu?cityId = 330100 & id = 330108 & type = district
    def requestComunityInfo(self):
        xiaoquList = []
        url = "http://hz.lianjia.com/xiaoqu/binjiang/"
        request = urllib2.Request(url, None, self.header)
        response = self.opener.open(request)
        data = response.read()
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
        return xiaoquList

    def getXiaoquInfo(self, url):
        time.sleep(random.randint(10, 100))
        xiaoquList = []
        request = urllib2.Request(url, None, self.header)
        response = self.opener.open(request)
        data = response.read()
        soup = BeautifulSoup(data, "html.parser")
        xiaoquTag = soup.find_all("li", attrs={'class': 'clear xiaoquListItem'})
        for tag in xiaoquTag:
            titleSet = tag.findAll("div", attrs={'class': 'title'})
            for title in titleSet:
                xiaoquUrl = title.a['href']
                xiaoquId = str(xiaoquUrl).split('/')[4]
                xiaoquName = title.a.contents[0]
                xiaoquList.append([xiaoquId, xiaoquName, xiaoquUrl])
        response.close()
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
