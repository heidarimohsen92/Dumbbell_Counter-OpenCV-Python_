import cv2 
import mediapipe as mp
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("dumbbell.mp4")
frame_size = np.int64((cap.get(3), cap.get(4)))
frame_rate = cap.get(5)
output = cv2.VideoWriter('counter.avi', cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, frame_size)

up = False
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        # mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        points = {}
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            points[id] = (cx,cy)

        cv2.circle(frame, points[11], 4, (255,0,0), cv2.FILLED)
        cv2.circle(frame, points[13], 4, (255,0,0), cv2.FILLED)
        cv2.circle(frame, points[15], 4, (255,0,0), cv2.FILLED)
        
        cv2.line(frame, points[11], points[13], (0,0,255), 2)
        cv2.line(frame, points[13], points[15], (0,0,255), 2)
        
        if not up and points[15][1]  > points[13][1]:
            up = True
            counter += 1
            print("UP")
        elif points[15][1] < points[13][1]:
            up = False
            print("Down")
        print("----------------------",counter)
        
    cv2.putText(frame, str(counter-1), (5,30),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    
    output.write(frame)
    
output.release()
cap.release()   
    