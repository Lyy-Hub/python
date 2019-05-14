def get_json(url, num):
   '''从网页获取JSON,使用POST请求,加上头部信息'''
   headers = {
       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36',
       'Host': 'www.lagou.com',
     'Referer':'https://www.lagou.com/jobs/list_python%E5%BC%80%E5%8F%91?labelWords=&;fromSearch=true&suginput=',
       'X-Anit-Forge-Code': '0',
     'X-Anit-Forge-Token': 'None',
     'X-Requested-With': 'XMLHttpRequest'
   }
   data = {
       'first': 'true',
       'pn': num,
       'kd': 'Python开发'}
   res = requests.post(url, headers=headers, data=data)
   res.raise_for_status()
   res.encoding = 'utf-8'
   # 得到包含职位信息的字典
   page = res.json()
   return page
