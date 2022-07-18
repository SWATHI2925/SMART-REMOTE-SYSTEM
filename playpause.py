import cv2
import numpy as np
import pyautogui
import time


while(1):

          

cv2.destroyAllWindows()



    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,48,80])
    upper_blue = np.array([20,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    blur = cv2.GaussianBlur(mask,(15,15),0)

    ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    if(len(contours))>=0:
        c=max(contours, key=cv2.contourArea)
        (x,y),radius=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
    else:print("Sorry no contour found")

    cnt=c
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    count=0
    try:
        defects.shape
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(frame,start,end,[0,255,0],2)
            cv2.circle(frame,far,5,[0,0,255],-1)
            count=count+1
        if cv2.arcLength(cnt,True)>2500:
            while ct==0:
                print("ON")
                pyautogui.press('space')
                ct=1
                ct1=0
        if cv2.arcLength(cnt,True)>1000 and cv2.arcLength(cnt,True)<=2000:
            while ct1==0:
                print("OFF")
                pyautogui.press('space')
                ct1=1
                ct=0          
    except AttributeError:print("shape not found")    
    cv2.imshow('final',frame)    