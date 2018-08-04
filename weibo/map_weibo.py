#!/share/apps/anaconda3/4.3.1/bin/python

### Reference: https://github.com/gaussic/weibo_wordcloud

import re
import json
import requests
import io

# based on mobile weibo
url_template = "https://m.weibo.cn/api/container/getIndex?type=wb&queryVal={}&containerid=100103type=2%26q%3D{}&page={}"


def clean_text(text):
    """cleaning"""
    dr = re.compile(r'(<)[^>]+>', re.S)
    dd = dr.sub('', text)
    dr = re.compile(r'#[^#]+#', re.S)
    dd = dr.sub('', dd)
    dr = re.compile(r'@[^ ]+ ', re.S)
    dd = dr.sub('', dd)
    return dd.strip()


def fetch_data(query_val, page_id):
    """crawl weibos based on specific keyword and page id"""
    resp = requests.get(url_template.format(query_val, query_val, page_id))
    card_group = json.loads(resp.text)['data']['cards'][0]['card_group']
    #print('url：', resp.url, ' --- num_weibos:', len(card_group))

    mblogs = []  # save them into dict
    for card in card_group:
        mblog = card['mblog']
        blog = {'mid': mblog['id'],  # weibo id
                'time': mblog['created_at'], # creation time
                'text': clean_text(mblog['text']),  # text
                'userid': str(mblog['user']['id']),  # user id
                'username': mblog['user']['screen_name'],  # username
                'reposts_count': mblog['reposts_count'],  # retweet
                'comments_count': mblog['comments_count'],  # comment
                'attitudes_count': mblog['attitudes_count']  # like
                }
        mblogs.append(blog)
    return mblogs


def fetch_pages(query_val, page_num):
    """crawl weibos on several pages"""
    
    for page_id in range(1 + page_num + 1):
        try:
            output = fetch_data(query_val, page_id)
            for j in output:
                print(str(j))
                
        except Exception as e:
            print(e)
            
fetch_pages('贵州茅台',30)