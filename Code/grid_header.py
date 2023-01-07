import cv2
import numpy as np
import cv2.aruco as aruco
import datetime
import socket
import time


''' ----- Variables -----   '''
# List of Detected Grid Coordinates (midPoints)
xcod = list()
ycod = list()
graph = {
    0: [16, 1],
    1: [2,17,0],
    2: [3,18,1],
    3: [4,19,2],
    4: [5,20,3],
    5: [6,21,4],
    6: [7,22,5],
    7: [8,23,6],
    8: [9,24,7],
    9: [10,25,8],
    10: [11,26,9],
    11: [12,27,10],
    12: [13,28,11],
    13: [14,29,12],
    14: [15,30,13],
    15: [31,14],
    16: [0,17],
    17: [18,1,16],
    18: [19,2,17],
    19: [20,3,18],
    20: [21,4,19],
    21: [22,5,20],
    22: [23,32,6,21],
    23: [24,33,7,22],
    24: [34,8],
    25: [26,35,9,24],
    26: [27,10,25],
    27: [28,11,26],
    28: [29,12,27],
    29: [30,13,28],
    30: [31,14,29],
    31: [15,30],
    32: [33,36,22],
    33: [34,37,23,32],
    34: [38,24],
    35: [39,25,34],
    36: [37,40,32],
    37: [38,41,33,36],
    38: [42,34],
    39: [43,38,35],
    40: [41,44,36],
    41: [42,45,37,40],
    42: [46,38],
    43: [47,39,42],
    44: [45,46,40],
    45: [46,49,41,44],
    46: [50,42],
    47: [51,43,46],
    48: [49,52,44],
    49: [50,53,45,48],
    50: [54,46],
    51: [55,47,50],
    52: [53,56,48],
    53: [54,57,49,52],
    54: [58,50],
    55: [59,51,54],
    56: [57,52],
    57: [58,53,56],
    58: [54],
    59: [55,58],
}

''' ----- Functions -----   '''
# Function to Plan a Path
def detectGrids(clicked, makeList):
    image = processedImg(clicked)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # To get center grid Coordinates of Each Detected Grids
    i = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        cv2.drawContours(clicked, [approx], 0, (0,255,0), 1)
        # x = approx.ravel()[0]
        # y = approx.ravel()[1]
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            x1 = int(x + h/2)
            y1 = int(y + w/2)
            print(str(h) + "," + str(w) + " -- " + str(aspectRatio))
            if aspectRatio >= 0.5 and aspectRatio <= 1.5 and h >= 20 and h <= 26 and makeList:
                cv2.putText(clicked, str(i), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255))
                xcod.append(x1)
                ycod.append(y1)
                i += 1
    return clicked
    # cv2.imshow("DetectedGrid", clicked)

def drawPath(image, graph, start, end, colour):
    pathPlanned = bfs(graph, start, end)
    pathCod = getCoords(pathPlanned)
    for index, item in enumerate(pathCod): 
        if index == len(pathCod) -1:
            break
        cv2.line(image, item, pathCod[index + 1], colour, 2)
    return image

def getCoords(pathPlanned):
    pathcods = []
    for p in pathPlanned:
        pathcods.append((xcod[p], ycod[p]))
    return pathcods

# Function to Process and improve Image for better Detection
def processedImg(rawImg):
    imgGray = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", imgGray)
    # blur = cv2.GaussianBlur(imgGray, (3,3), 0)
    # cv2.imshow("blur", blur)
    # edges = cv2.Canny(blur, 100, 255, apertureSize=3)
    # cv2.imshow('edges', edges)
    _, thrash = cv2.threshold(imgGray, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thrash)
    return thrash

def bfs(graph_to_search, start, end):
    queue = [[start]]
    visited = set()

    while queue:
        # Gets the first path in the queue
        path = queue.pop(0)

        # Gets the last node in the path
        vertex = path[-1]

        # Checks if we got to the end
        if vertex == end:
            return path
        # We check if the current node is already in the visited nodes set in order not to recheck it
        elif vertex not in visited:
            # enumerate all adjacent nodes, construct a new path and push it into the queue
            for current_neighbour in graph_to_search.get(vertex, []):
                new_path = list(path)
                new_path.append(current_neighbour)
                queue.append(new_path)

            # Mark the vertex as visited
            visited.add(vertex)

def showDateTime(frame):
    dateTime = str(datetime.datetime.now())
    frame = cv2.putText(frame, dateTime, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

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

def pidControll():
    pass

def setWiFi(ip_address, port, max_connect=4):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip_address, port))
    s.listen(max_connect)

def wifiCommunication():
    pass