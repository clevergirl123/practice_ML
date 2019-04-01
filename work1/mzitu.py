# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import os

headers = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"}

all_url = 'http://www.mzitu.com/all'

start_html = requests.get(all_url, headers = headers)

#print(start_html.text)
#把这个网页的代码都打下来

Soup = BeautifulSoup(start_html.text, 'lxml')

all_a = Soup.find('div', class_='all').find_all('a')


for a in all_a:
	title = a.get_text()
	href = a['href']
#	print(title, href)
	html = requests.get(href,headers = headers)

	html_Soup = BeautifulSoup(html.text,'lxml')

	max_span = html_Soup.find('div',class_='pagenavi').find_all('span')[-2].get_text()

	for page in range(1, int(max_span) + 1):
		page_url = href + '/' + str(page)
#		print(page_url) 
		img_html = requests.get(page_url, headers = headers)

		img_Soup = BeautifulSoup(img_html.text, 'lxml')

		img_url = img_Soup.find('div', class_ = 'main-image').find('img')['src']

#		print(img_url)

		name = img_url[-9: -4]

		img = requests.get(img_url, headers = headers)

		f = open(name + '.jpg', 'ab')

		f.write(img.content)

		f.close()