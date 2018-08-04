from bs4 import BeautifulSoup
import requests
import sys
import io

def get_news_titles(keyword):
    with io.open('{}_baidu.txt'.format(keyword),'w',encoding='utf-8') as g:
        keyword = keyword
        page_id = 1
        ids = []
        while True:
            url = 'http://news.baidu.com/ns?word=title%3A%28{}%29&pn={}&cl=2&ct=0&tn=newstitledy&rn=50&ie=utf-8&bt=1420041600&et=1514822399'.format(keyword,0+50*(page_id-1))
            r = requests.get(url)
            html = r.text
            soup = BeautifulSoup(html, 'html.parser') 
            div_items = soup.find_all('div', class_='result') 
            for i in div_items:
                # get titles
                title = i.contents[1].find('a').contents
                title = [x.string for x in title]
                title = ''.join(title).strip()
                # get publish time
                time = i.contents[3].contents[0].string.strip().split('\t')[-1]
                id_pos = str(i).find('id')
                # get news id
                id = str(i)[id_pos+4:id_pos+7].replace('"','').replace('>','')
                if id in ids:
                    print('finished!!!')
                    return
                ids.append(id)
                output = str({'id':id,'title':title,'time':time})
                g.write(output)
                g.write('\n')
                print('id: {}'.format(id))
                print('title: {}'.format(title))
                print('time: {}'.format(time))
            page_id += 1
            print('now on page: {}'.format(page_id))
        
if __name__ == '__main__':
    get_news_titles(sys.argv[1])