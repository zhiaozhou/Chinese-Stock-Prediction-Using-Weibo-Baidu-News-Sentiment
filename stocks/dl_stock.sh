#!/usr/bin/env bash

python get_stock.py
hdfs dfs -put stock_price.xlsx stock/
