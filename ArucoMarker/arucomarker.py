# https://www.youtube.com/watch?v=v5a7pKSOJd8

import cv2
import cv2.aruco as aruco
import numpy as np
import math
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

        C = ( int((c1[0]+c2[0]+c3[0]+c4[0])/4) , int((c1[1]+c2[1]+c3[1]+c4[1])/4) ) # Center
        F = ( int((int(c1[0]) + int(c2[0]))/2) , int((int(c1[1]) + int(c2[1]))/2) ) # Front
        L = ( int((int(c1[0]) + int(c4[0]))/2) , int((int(c1[1]) + int(c4[1]))/2) ) # Left
        R = ( int((int(c2[0]) + int(c3[0]))/2) , int((int(c2[1]) + int(c3[1]))/2) ) # Right

        length = 100
        
        # FRONT
        lenCF = math.sqrt(math.pow(C[0] - F[0], 2.0) + math.pow(C[1] - F[1], 2.0))
        Xf = int(F[0] + (F[0] - C[0]) / lenCF * length)
        Yf = int(F[1] + (F[1] - C[1]) / lenCF * length)
        img = cv2.arrowedLine(img, (C[0], C[1]), (Xf, Yf), (0,0,255), 2)

        # LEFT
        lenCL = math.sqrt(math.pow(C[0] - L[0], 2.0) + math.pow(C[1] - F[1], 2.0))
        Xl = int(L[0] + (L[0] - C[0]) / lenCL * length)
        Yl = int(L[1] + (L[1] - C[1]) / lenCL * length)
        img = cv2.arrowedLine(img, (C[0], C[1]), (Xl, Yl), (0,255,0), 2)

        # RIGHT
        lenCR = math.sqrt(math.pow(C[0] - R[0], 2.0) + math.pow(C[1] - R[1], 2.0))
        Xr = int(R[0] + (R[0] - C[0]) / lenCR * length)
        Yr = int(R[1] + (R[1] - C[1]) / lenCR * length)
        img = cv2.arrowedLine(img, (C[0], C[1]), (Xr, Yr), (255,0,0), 2)

        img = cv2.putText(img, str(ids[i]), (C[0], C[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
imgAug = cv2.imread('pixio.jpg')
while True:
    SUCCESS, img = cap.read()
    arucoFound = findRrucoMarkers(img)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):       # Press 'q' to Exit.......
        break

cap.release()
cv2.destroyAllWindows()