#!/usr/bin/env python
# -*-coding:utf-8 -*-
# ** author:toby
# ** description: more function
# ** run python: python xxxx.py param1 param2
# ********************
import requests
import jieba
import re
from xpinyin import Pinyin
p = Pinyin()

RhymeIndex = [('1', ['a', 'ia', 'ua']), ('2', ['ai', 'uai']), ('3', ['an', 'ian', 'uan']),
              ('4', ['ang', 'iang', 'uang']), ('5', ['ao', 'iao']), ('6', ['e', 'o', 'uo']), ('7', ['ei', 'ui']),
              ('8', ['en', 'in', 'un']), ('9', ['eng', 'ing', 'ong', 'iong']), ('10', ['er']), ('11', ['i']),
              ('12', ['ie', 'ye']), ('13', ['ou', 'iu']), ('14', ['u']), ('16', ['ue']), ('15', ['qu', 'xu', 'yu'])]

RhymeDct = {'ui': '7', 'uan': '3', 'ian': '3', 'iu': '13', 'en': '8', 'ue': '16', 'ing': '9', 'a': '1', 'ei': '7',
            'eng': '9', 'uo': '6', 'ye': '12', 'in': '8', 'ou': '13', 'ao': '5', 'uang': '4', 'ong': '9', 'ang': '4',
            'ai': '2', 'ua': '1', 'uai': '2', 'an': '3', 'iao': '5', 'ia': '1', 'ie': '12', 'iong': '9', 'i': '11',
            'er': '10', 'e': '6', 'u': '14', 'un': '8', 'iang': '4', 'o': '6', 'qu': '15', 'xu': '15', 'yu': '15'}


def _analysis_words(words):
    word_py = p.get_pinyin((u'{}'.format(words)))
    lst_words = word_py.split('-')
    r = []
    for i in lst_words:
        while True:
            if not i:
                break
            token = RhymeDct.get(i, None)
            if token:
                r.append(token)
                break
            i = i[1:]
    if len(r) == len(words):
        return '-'.join(r)
# print(_analysis_words('兄弟'))
def GetKeyword():
    # 歌曲列表
    # url = 'http://music.163.com/api/playlist/detail?id=808976784'
    # req = requests.get(url)
    # data = req.json()
    # print(data['result']['tracks'] )
    # tracks =data['result']['tracks']  #歌曲列表
    tracks = ["431795900", '33850315', '430053482']
    # 写入记事本文件
    with open('keyword.txt', 'a') as f:
        f.write("[")
        for i in tracks:
            print(111)
            # 歌词    "http://music.163.com/api/song/lyric?os=pc&id=431795900&lv=-1&kv=-1&tv=-1"
            # lrcurl = "http://music.163.com/api/song/lyric?os=pc&id="+str(i['id'])+"&lv=-1&kv=-1&tv=-1"
            lrcurl = "http://music.163.com/api/song/lyric?os=pc&id=" + str(i) + "&lv=-1&kv=-1&tv=-1"
            lrcreq = requests.get(lrcurl)
            dt = lrcreq.json()
            lrc = re.sub(u"\\[.*?]", "", dt['lrc']['lyric'])
            # jieba分词
            seg_list = list(jieba.cut(lrc, cut_all=True))
            for i in seg_list:
                # 加入判断，只写入2个字组成的词
                if len(i) == 2:
                    # 写入格式：{'7-13':'追求'}
                    if _analysis_words(i) != None:
                        f.write("{'" + _analysis_words(i) + "':'" + i + "'},")
        f.write("]")
        f.close()


def Findkey(str):
    result={}
    with open('keyword.txt', 'r') as f:
        # print(f.readlines())
        list=eval(f.readlines()[0])
        for item in list:
            if item.get(str):
                key=item.get(str)
                number=result.get(key)
                #如果一个词出现多次，进行次数累加，用来表示频次
                if number !=None and number>=1:
                    result[key]=number+1
                else:
                    result.update({key:1})
        f.close()
        print(result)

if __name__=="__main__":
    GetKeyword()
    key = input("请输入关键词:")
    str = _analysis_words(key)
    print("匹配押韵的词：")
    Findkey(str)