#!/usr/bin/env bash

python get_news_title.py 贵州茅台 10
hdfs dfs -put 贵州茅台_baidu.txt baidu/
