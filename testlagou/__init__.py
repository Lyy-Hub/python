position_url = []
def read_csv():
    # 读取文件内容
    with open(r'D:\\lagou_1.csv', 'r', newline='') as file_test:
        # 读文件
        reader = csv.reader(file_test)
        i = 0
        for row in reader:
            if i != 0 :
                # 根据positionID补全链接
                url_single = "https://www.lagou.com/jobs/%s.html"%row[0]
                position_url.append(url_single)
            i = i + 1
        print('一共有：'+str(i-1)+'个')
        print(position_url)