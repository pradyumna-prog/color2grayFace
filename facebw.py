import cv2
import os
from os.path import isfile, join

faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def onphoto(img):
    imgName = img.split("\\")[-1]
    frame = cv2.imread(img)
    frame = cv2.resize(frame, (0, 0), fx = 0.1, fy = 0.1)
    faces = faceClassifier.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30)
    )
    for (x,y,w,h) in faces:        
        only_face = frame[y: y+h, x: x+w]
        grayface = cv2.cvtColor(only_face, cv2.COLOR_BGR2GRAY)
        frame[y: y+h, x: x+w] = cv2.merge([grayface, grayface, grayface])
        cv2.imwrite("F:\\grayImages\\"+imgName, frame)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
        #cv2.imshow("Only Gray Face", frame)

    cv2.waitKey(0)

def liveCapture():
    cap  = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        faces = faceClassifier.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
            only_face = frame[y: y+h, x: x+w]
            grayface = cv2.cvtColor(only_face, cv2.COLOR_BGR2GRAY)
            frame[y: y+h, x: x+w] = cv2.merge([grayface, grayface, grayface])

        cv2.imshow("Only Gray Face", frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    liveCapture()
    '''
    path = "E:\\Photos\\ours"
    imgArr = [join(path, img) for img in os.listdir(path) if isfile(join(path, img))]
    for img in imgArr:
        try:
            onphoto(img)
        except Exception as e:
            print(e)
    '''