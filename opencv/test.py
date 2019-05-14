#打开摄像头
import cv2
#摄像头对象
cap=cv2.VideoCapture(0)
#显示
while(1):
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = cv2.Canny(img_gray, 100 , 100)

    # temp = cv2.blur(temp, (9, 9))#模糊
    temp = cv2.bitwise_not(temp,(1,1))#变白

    cv2.imshow("capture", temp)
    if(cv2.waitKey(1) & 0xFF==ord(' ')):#按空格退出
        break
cap.release()
cv2.destroyAllWindows()
