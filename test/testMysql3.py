# encoding=utf-8
import mysql.connector

db = mysql.connector.connect(host="localhost", user="root", passwd="root", database='dbgirl')
cursor = db.cursor()

# 删除Students表中Name=ZG的数据
#cursor.execute("DELETE FROM user WHERE id='1'")

count = cursor.execute("SELECT * FROM user")
print("总共有" + str(count) + "条记录")
rows = cursor.fetchall()
for row in rows:
    print(row[0], row[1])
db.commit()
db.close()