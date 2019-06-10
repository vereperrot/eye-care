"""
python -m pip install opencv-python
python camera.py haarcascade_frontalface_alt.xml
"""
import cv2,sys,time
#import sys
from functools import reduce


def listAllAvailableVideoDevice():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr
if len(listAllAvailableVideoDevice())==0:
    print('Warning: video source not found')
    sys.exit()
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
video_capture = cv2.VideoCapture(0)

eyes_count=[]
open_eyes=0
close_eyes=0
rec=False
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Change the eye haarcascade for better recognization
        if len(eyes_count)<=15:
            eyes_count.append(len(eyes))
        if len(eyes_count)==15:
            avg=reduce(lambda x, y: x + y, eyes_count) / len(eyes_count)
            print(avg)
            if avg>2:
                print("change haarcascade to eyeglasses")
                eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
            if rec==False:
                start = time.time()
                rec=True
            if rec:
                done=time.time()
                elapsed = done - start
                if elapsed>=20*60:
                    print("open_eyes:"+str(open_eyes))
                    print("close_eyes:"+str(close_eyes))
                    time.sleep(20)#secs
                    break
            if len(eyes)==0: 
                close_eyes=close_eyes+1
            else:
                open_eyes=open_eyes+1
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
