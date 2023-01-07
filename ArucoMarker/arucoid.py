import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findRrucoMarkers(img, markerSize=4, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    #print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    
    for (i,b) in enumerate(bboxs):
        c1 = (b[0][0][0],b[0][0][1])
        c2 = (b[0][1][0],b[0][1][1])
        c3 = (b[0][2][0],b[0][2][1])
        c4 = (b[0][3][0],b[0][3][1])
        x = int((c1[0]+c2[0]+c3[0]+c4[0])/4)
        y = int((c1[1]+c2[1]+c3[1]+c4[1])/4)
        x1 = int((int(c1[0]) + int(c2[0]))/2)
        y1 = int((int(c1[1]) + int(c2[1]))/2)
        x2 = int((int(c3[0]) + int(c4[0]))/2)
        y2 = int((int(c3[1]) + int(c4[1]))/2)
        img = cv2.putText(img, str(ids[i]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA) 
        img = cv2.arrowedLine(img, (x2,y2), (x1,y1), (0,0,255), 2)
    return [bboxs, ids]

cap = cv2.VideoCapture(0)
while True:
    SUCCESS, img = cap.read()
    arucoFound = findRrucoMarkers(img)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):       # Press 'q' to Exit.......
        break

cap.release()
cv2.destroyAllWindows()