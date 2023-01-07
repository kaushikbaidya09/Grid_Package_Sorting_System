import cv2
import numpy as np
import cv2.aruco as aruco
import grid_header as rc

'''  ---  SET UP VALUES  ---  '''
makeList = True     ## Condition to creat a list of detected grid Coordinates
video = True        ## True -- work with camera     ## False -- work with image
width = 500         ## SET VIDEO RESOLUTION
height = 270
RecordVideo = False
ip_address = '192.168.43.221'
port = 9999

rc.setWiFi(ip_address, port)

if video:
    cap = cv2.VideoCapture(0)
    cap.set(3,width)        # index 3 for width OR cv2.CAP_PROP_FRAME_WIDTH
    cap.set(4,height)       # index 4 for hight OR cv2.CAP_PROP_FRAME_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWrite('RaceVideo.avi', fourcc, 20.0, (640,480))
    # SAVE VIDEO CAPTURE(FILE_NAME, Variable, Frame_Rate, Frame dimentions)

    # Camera Checking
    while True:
        _, frame = cap.read()
        gridimg = rc.detectGrids(frame, False)
        cv2.imshow("webCam", gridimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):       # Press 'q' to proceed.......
            cv2.destroyAllWindows()
            break

    # This takes a picture from the camera
    if cap.isOpened():
        _, clicked = cap.read()
    cv2.imshow("CapturedIMG", clicked)

    clicked = cv2.imread('Arena500x270.png')
    gridimg = rc.detectGrids(clicked, makeList)
    cv2.imshow("DetectedImage", gridimg)

    # Draw Path on VIDEO
    while True:
        _, frame = cap.read()
        rc.findRrucoMarkers(frame, 4, 250)
        cv2.imshow("DrawnPathVIDEO", frame)
        if makeList:
            imgpath = rc.drawPath(frame, rc.graph, 59, 31, [0,255,255])
            imgpath = rc.drawPath(frame, rc.graph, 58, 15, [255,0,0])
            imgpath = rc.drawPath(frame, rc.graph, 57, 0, [0,0,255])
            imgpath = rc.drawPath(frame, rc.graph, 56, 16, [255,255,0])
            cv2.imshow("DrawnPathVIDEO", imgpath)
        rc.showDateTime(frame)

        if RecordVideo:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):       # Press 'q' to Exit.......
            break
    
    cap.release()
    cv2.destroyAllWindows()

else:
    clicked = cv2.imread('Arena500x270.png')
    photo = cv2.imread('Arena500x270.png')
    gridimg = rc.detectGrids(clicked)
    cv2.imshow("DetectedImage", gridimg)

    # Draw Path on IMAGE
    imgpath = rc.drawPath(photo, rc.graph, 59, 31, [0,255,255])
    imgpath = rc.drawPath(photo, rc.graph, 58, 15, [255,0,0])
    imgpath = rc.drawPath(photo, rc.graph, 57, 0, [0,0,255])
    imgpath = rc.drawPath(photo, rc.graph, 56, 16, [255,255,0])
    cv2.imshow("DrawnPathIMAGE", imgpath)

    cv2.waitKey(0)
    cv2.destroyAllWindows()