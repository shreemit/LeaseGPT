import time
# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
# import re
# import urllib.request
# # from urlparse import urljoin

# req = Request("https://seattle.craigslist.org/search/apa#search=1~gallery~0~0",
#                headers={'User-Agent': 'Mozilla/5.0'})
# html_page = urlopen(req)

# soup = BeautifulSoup(html_page, "html.parser")
# # print(soup.body)
# links = []
# for link in soup.find_all('a'):
#     links.append(link)

# print(links)

# import pandas as pdcl
# from bs4 import BeautifulSoup
# import cloudscraper
# scraper = cloudscraper.create_scraper()
# kw= ['Oxford', 'Oxfordshire']
# data = []
# for k in kw:
#     for page in range(1,3):
#         url = f"https://www.zoopla.co.uk/for-sale/property/oxford/?search_source=home&q={k}&pn={page}"
#         page = scraper.get(url)
#         #print(page)
#         soup = BeautifulSoup(page.content, "html.parser")
    
#         for card in soup.select('[data-testid="regular-listings"] [id^="listing"]'):
#             print(card.a.get("href"))
#             link = "https://www.zoopla.co.uk" + card.a.get("href")
#             print(link)
#             #data.append({'link':link})

from selenium import webdriver
from bs4 import BeautifulSoup
browser = webdriver.Firefox()
options = webdriver.FirefoxOptions()
options.add_argument('--enable-javascript')
options.add_argument("--headless")
browser.get("https://seattle.craigslist.org/search/apa#search=1~gallery~0~0")
time.sleep(1)
html = browser.page_source
soup = BeautifulSoup(html, 'html.parser')
# Find all the links on the page from galary inner
L = soup.find_all('a', class_='main')
time.sleep(1)
browser.get(L[0]['href'])
soup2 = BeautifulSoup(browser.page_source, 'html.parser')
print(soup2.find_all('section', class_='body'))