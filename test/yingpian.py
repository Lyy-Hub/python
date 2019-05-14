import csv
import requests
from bs4 import BeautifulSoup  # 从bs4引入BeautifulSoup

# 请求网页
url = "https://movie.douban.com/cinema/later/chengdu/"
response = requests.get(url)
# 初始化BeautifulSoup方法一：利用网页字符串自带的编码信息解析网页
soup = BeautifulSoup(response.content.decode('utf-8'), 'lxml')

# 初始化BeautifulSoup方法二：手动指定解析编码解析网页
# soup = BeautifulSoup(response.content, 'lxml', from_encoding='utf-8')

# print(soup)  # 输出BeautifulSoup转换后的内容
all_movies = soup.find('div', id="showing-soon")  # 先找到最大的div
# print(all_movies)  # 输出最大的div的内容

csv_file = open('data.csv', 'w', encoding="gbk", newline='')
writer = csv.writer(csv_file)

writer.writerow(["影片名", "链接", "上映日期", "影片类型", "地区", "关注者"])  # 写入标题
for each_movie in all_movies.find_all('div', class_="item"):  # 从最大的div里面找到影片的div
    # print(each_movie)  # 输出每个影片div的内容
    all_a_tag = each_movie.find_all('a')
    all_li_tag = each_movie.find_all('li')
    movie_name = all_a_tag[1].text
    moive_href = all_a_tag[1]['href']
    movie_date = all_li_tag[0].text
    movie_type = all_li_tag[1].text
    movie_area = all_li_tag[2].text
    movie_lovers = all_li_tag[3].text.replace("想看", '')
    print('名字：{}，链接：{}，日期：{}，类型：{}，地区：{}， 关注者：{}'.format(
        movie_name, moive_href, movie_date, movie_type, movie_area, movie_lovers))
    writer.writerow([movie_name, moive_href, movie_date, movie_type, movie_area, movie_lovers])

csv_file.close()
print("write_finished!")