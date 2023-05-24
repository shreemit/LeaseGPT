# work with dataframes
import pandas as pd
import numpy as np

# we'll use this to store the a dataframe to csv at a later stage
import csv

# make HTTP requests to a specified URL
import requests

# web scraping library
from bs4 import BeautifulSoup

# time management
import time
from random import randint

from tqdm import tqdm
# from collections import Counter

# regular expressions
import re

# a link to each vehicle post will be stored here
links = []

#  all pages in a city will be stored in this list
list_of_cities = []

# list of city names we want to get data for
cities = ['dallas','chicago', 'newyork', 'sfbay', 'losangeles', \
        'houston', 'phoenix', 'philadelphia', 'sanantonio', 'washingtondc',\
       'boston', 'nashville', 'atlanta', 'miami', 'seattle']


for city in cities:
    # each city has approxiamtely 1800 pages for the "cars for sale by owner" category
    # we'll keep track of these pages with page_number variable below
    page_number = 1

    # this while loop cycles through all 1800 pages
    while page_number <= 1800:
        # city_link variable takes a a different city name from the cities every time through the loop
        city_link = "https://" + str(city) + ".craigslist.org/d/cars-trucks-by-owner/search/cto?s=" + \
                                str(page_number) + "&hasPic=1"

        # we                         
        list_of_cities.append(city_link)
        page_number +=120

# URLs counter
car_urls = 1      
for each_city_page in list_of_cities:
    links_in_each_city_page = requests.get(each_city_page)
    # parse html object from page to BS object
    soup = BeautifulSoup(links_in_each_city_page.content, 'html.parser')

    try:
        #get the macro-container for the car posts for that page
        posts = soup.find_all('a', class_= 'result-image gallery')

        # get all the html links in the page and append them to a list
        for link in tqdm(posts):
            l = link.get('href')
            links.append(l)

    except:
        pass

    if car_urls % 5 == 0:
        city_link = l.strip()
        start = city_link.find("//") + len("//")
        end = city_link.find(".")
        city_string = city_link[start:end]
        print('Number of pages returned --> ' + str(car_urls) + '---' + city_string)
            

    # this code just helps us keep count of the looping progress                     


    # we add a sleep timer to manage our server requests
    time.sleep(randint(0,1))
    car_urls +=1
print('finished scraping all the links')