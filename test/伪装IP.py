import urllib.request
url = r'http://douban.com'
headers = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0'}

#urllib.request.Request（）用于向服务端发送请求，就如 http 协议客户端向服务端发送请求 POST
#添加了一个头部，伪装成浏览器,此时的url并不是一个裸露的url，而是具有header头部的url
req = urllib.request.Request(url=url, headers=headers)

#urllib.request.urlopen（）则相当于服务器返回的响应,返回的是一个request类的一个对象， GET
# 类似于一个文件对象，可以进行ｏｐｅｎ()操作获取内容
res = urllib.request.urlopen(req)

html = res.read().decode('utf-8')
res.close()
print(html)