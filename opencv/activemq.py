# -*-coding:utf-8-*-
import stomp
import time
import base64
import random
import string

queue_name = '/queue/test'
listener_name = 'SampleListener'

class SampleListener(object):
    def on_error(self, headers, message):
        print('错误：', message)
    def on_message(self, headers, message):
        print('接收：', message)
        #  随机字符串
        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        img = base64.b64decode(message)
        file = open('E:/OpenCV/img/'+ ran_str +'.jpg', 'wb')
        file.write(img)
        file.close()

# 推送到队列queue
def send_to_queue(msg):
    conn = stomp.Connection10([('127.0.0.1', 61613)])
    conn.start()
    conn.connect()
    print('发送：', msg)
    conn.send(queue_name, msg)
    conn.disconnect()

##从队列接收消息
def receive_from_queue():
    conn = stomp.Connection10([('127.0.0.1', 61613)])
    conn.set_listener(listener_name, SampleListener())
    conn.start()
    conn.connect()
    conn.subscribe(queue_name)
    time.sleep(1)  # secs
    conn.disconnect()


if __name__ == '__main__':
     #send_to_queue('len 123')
     # 无限循环，接收队列中的消息
     var = 1
     while var == 1:
         receive_from_queue()
