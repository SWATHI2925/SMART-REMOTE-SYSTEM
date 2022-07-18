import cv2
import math
import time
import pyautogui
import numpy as np
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from Modules import HandTrackingModule as htm
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#VOLUME  
wCam,hCam,cap,pTime = 640,480,cv2.VideoCapture(0),0
cap.set(3, wCam)
cap.set(4, hCam)
 
isVideoInPause = False

detector = htm.handDetector(detectionCon=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol,maxVol,vol,volBar,volPer = volRange[0],volRange[1],0,400,0

#PAUSE PLAY
ct,ct1 = 0,0

def checkForPlayPause():
    global ct,ct1,img,isVideoInPause
    frame = img
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
                isVideoInPause = True
                pyautogui.press('space')
                ct=1
                ct1=0
        if cv2.arcLength(cnt,True)>1000 and cv2.arcLength(cnt,True)<=2000:
            while ct1==0:
                print("OFF")
                isVideoInPause = False
                pyautogui.press('space')
                ct1=1
                ct=0          
    except AttributeError:print("shape not found")    
    img = frame    

def checkForVolume():
    global wCam,hCam,cap,pTime,detector,devices,interface,volume,volRange,minVol,maxVol,vol,volBar,volPer,img
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [50, 200], [minVol, maxVol])
        volBar = np.interp(length, [50, 200], [400, 150])
        volPer = np.interp(length, [50, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)

if(__name__ == "__main__"):
    while(True):
        _, img = cap.read()

        if(isVideoInPause): checkForVolume()            
        checkForPlayPause()

        cv2.imshow("Img", img)
        cv2.waitKey(1)