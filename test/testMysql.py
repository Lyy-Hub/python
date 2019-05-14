#!/usr/bin/python
# -*- coding: UTF-8 -*-

import MySQLdb

# 打开数据库连接
db = MySQLdb.connect("localhost", "root", "root", "dbgirl", charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()
sql = "delete from test where id = '1'"
try:
    cursor.execute(sql)
    db.commit()
except:
    db.rollback()

# 关闭数据库连接
cursor.close()
db.close()