import requests
import re

import mysql.connector
import csv
import time
import pandas as pd
from bs4 import BeautifulSoup
from requests import RequestException

#连接数据库
conn= mysql.connector.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='root',
        db ='dbgirl',  #数据库表 如果没有这个表则注释 使用执行语句建立一数据库
        charset='utf8',#设置编码集   如果没有则会报错UnicodeEncodeError: 'latin-1' codec can't encode characters in position 0-2: ordinal not in range(256)
        )
cur = conn.cursor() #要想使用数据库则还需要创建游标

#创建表
cur.execute("create table wdtest(wid int primary key auto_increment,name varchar(10),total int,rate varchar(10),pnum int,cycle varchar(10),p1num int,fuload varchar(10),alltotal varchar(10),capital varchar(10))")

#抓取页面
def get_page(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari / 537.11',
                              'Accept':'text/html;q=0.9,*/*;q=0.8',
                             'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                             'Connection': 'close',
                             'Referer': 'https://www.bluewhale.cc/'
    }
    try:
        response = requests.get(url,headers=headers)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        print('请求错误')
        return None

#解析页面  并写入数据
def parse_page(html):
    soup = BeautifulSoup(html,'lxml')
    tr = soup.find_all(name='tr',attrs={'class':'bd'})
    # tr = str(tr)
    for i in tr:
        # print(i)
        i = str(i)
        num = re.findall('<td class="num">(.*?)</td>',i)
        name = re.findall('.*?<a.*?title="(.*?)">',i)
        total = re.findall('<td class="total">(.*?)万</td>',i)
        rate = re.findall('<td class="rate">(.*?)%</td>',i)
        pnum = re.findall('<td class="pnum">(\d+)人</td>',i)
        cycle = re.findall('<td class="cycle">(.*?)月</td>',i)
        p1num = re.findall('<td class="p1num">(.*?)人</td>',i)
        fuload = re.findall('<td class="fuload">(.*?)分钟</td>',i)
        alltotal = re.findall('<td class="alltotal">(.*?)万</td>',i)
        capital = re.findall('<td class="capital">(.*?)万</td>',i)
        print(num,name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital)
        # 写入csv数据
        write_csv(num,name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital)
        #写入数据库  因为是没获取一天数据插入数据库，这样操作太耗时！可以mysql的考虑批处理！
        save_msg_toMySql(name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital)
        # print( tr)

#写入csv数据
def write_csv(num,name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital):
    columns = ['平台名称', '成交额(万)', '综合利率', '投资人(人)', '借款周期(月)', '借款人(人)', '满标速度(分钟)', '累计贷款余额(万)','净资金流入(万)']
    table = pd.DataFrame({'平台名称': name,
                          '成交额(万)': total,
                          '综合利率': rate,
                          '投资人(人)': pnum,
                          '借款周期(月)': cycle,
                          '借款人(人)': p1num,
                          '满标速度(分钟)': fuload,
                          '累计贷款余额(万)': alltotal,
                          '净资金流入(万)': capital},
             columns=columns)
    table.to_csv('wangdai.csv',mode='a',index=False,header=False) #不要索引 不要列头

#生成csv文件列头
def write_csv_lietou():
    columns = ['平台名称', '成交额(万)', '综合利率', '投资人(人)', '借款周期(月)', '借款人(人)', '满标速度(分钟)', '累计贷款余额(万)','净资金流入(万)']
    table = pd.DataFrame(columns=columns)
    table.to_csv('wangdai.csv',mode='a',index=False) #不要索引  是在第一行和第一列留一行

#保存数据到MySql
def save_msg_toMySql(name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital):
    sql = 'insert into wdtest(name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)'
    cur.execute(sql,(name,total,rate,pnum,cycle,p1num,fuload,alltotal,capital))
    #提交  记得commit一下，否则你的数据都没有到数据库哦
    conn.commit()


#主函数
if __name__ == '__main__':
    url = 'http://www.p2peye.com/shuju/ptsj/'
    html = get_page(url)
    #生成含有csv的列头
    write_csv_lietou()
    parse_page(html)