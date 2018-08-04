import requests
requests.adapters.DEFAULT_RETRIES = 1000
import glob
import pandas as pd
from datetime import datetime
import io
import re
from cucco import Cucco
from unidecode import unidecode
import pandas as pd
from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = '11584634'
API_KEY = 'ulwuucx5BFFdcKX1xn3P4y2C'
SECRET_KEY = 'DV3HxNeorC4gEno1qmrW1lyxZN18B9V9'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

def remove_emoji(text):
    cucco = Cucco()
    return cucco.replace_emojis(text)

def is_ustr(in_str):
    """transfer non-chinese unicodes to utf-8"""
    out_str=''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str=out_str+in_str[i]
        else:
            out_str=out_str+unidecode(in_str[i])
    return out_str
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    #"""判断一个unicode是否是数字"""
    #if uchar >= u'\u0030' and uchar<=u'\u0039':
    #        return False        
    #"""判断一个unicode是否是英文字母"""
    #if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
    #        return False
    #if uchar in ('-',',','，','。','.','>','?'):
    #        return False
    return False

def get_sentiment(text):
    text = is_ustr(remove_emoji(text))
    sentiment_result = client.sentimentClassify(text)
    sentiment = sentiment_result['items'][0]['sentiment']
    confidence = sentiment_result['items'][0]['confidence']
    return [sentiment,confidence]
