#!/usr/bin/python
# -*- coding: UTF-8 -*-
import smtplib
from email.mime.text import MIMEText
from email.header import Header

#sender = '13953193651@163.com'
#receiver = '1095285545@qq.com'
#subject = '测试'
#smtpserver = 'smtp.163.com'
#username = '13953193651@163.com'
#password = 'liyueyang111'

# 用qq邮箱往163邮箱发送邮件验证
sender = '1095285545@qq.com' #发送方  邮箱地址
receiver = '1095285545@qq.com'#接收方  邮箱地址
subject = '测试' #接收方  邮箱标题
smtpserver = 'smtp.qq.com'#发送方  邮箱协议SMTP
username = '1095285545@qq.com'#发送方  邮箱地址
password = 'cydhhfmfryjrbacg'#发送方 邮箱授权码

msg = MIMEText( '牛逼北咨', 'text', 'utf-8' ) # 接收方  邮箱内容
msg['Subject'] = Header( subject, 'utf-8' )# 接收方  邮箱头

smtp = smtplib.SMTP()
smtp.connect( smtpserver )
smtp.login( username, password )
smtp.sendmail( sender, receiver, msg.as_string() )
smtp.quit()