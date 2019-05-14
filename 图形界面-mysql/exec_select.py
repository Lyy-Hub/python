# 导入访问MySQL的模块
import mysql.connector

def query_db():
    # ①、连接数据库
    conn = conn = mysql.connector.connect(user='root', password='root',
        host='localhost', port='3306',
        database='aid', use_unicode=True)
    # ②、获取游标
    c = conn.cursor()
    # ③、调用执行select语句查询数据
    c.execute('select * from user_t')
    # 通过游标的description属性获取列信息
    description = c.description
    # 使用fetchall获取游标中的所有结果集
    rows = c.fetchall()
    # ④、关闭游标
    c.close()
    # ⑤、关闭连接
    conn.close()
    return description, rows