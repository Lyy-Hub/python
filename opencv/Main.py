import base64
import os
import stomp
import cv2

queue_name = '/queue/test'
listener_name = 'SampleListener'
# 推送到队列queue
def send_to_queue(msg):
    conn = stomp.Connection10([('10.10.39.17', 61613)])
    conn.start()
    conn.connect()
    print('发送：', msg)
    conn.send(queue_name, msg)
    conn.disconnect()

cap = cv2.VideoCapture(0)
i = 0
while (1):
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord(' '):
        # 写入本地
        cv2.imwrite(os.getcwd() + '/' + str(i) + '.jpg', frame)
        # 读取本地图片，转成base64
        with open(os.getcwd() + '/' + str(i) + '.jpg', "rb") as f:
            base64_data = base64.b64encode(f.read())
            print(base64_data)
            # 将base64编码，发送到activemq队列中
            send_to_queue(base64_data)
        i += 1
    cv2.imshow("capture", frame)
cap.release()