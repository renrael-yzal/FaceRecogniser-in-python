import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
import pandas as pd
labels=pd.read_csv("classes_indices.csv")
labels.set_index("Class")
def predictor(count=10000):
        model=load_model("face_Detector.h5")
        face_detector=MTCNN()
        print("models and modules loaded")
        cam=cv2.VideoCapture(0)
        while count>0:
                #print("loop is running")
                ret,frame=cam.read()
                faces=face_detector.detect_faces(frame)
                if len(faces)>0:
                    x,y,w,h=faces[0]["box"]
                    #frame=cv2.rectangle(frame,(w,h),(x,y),(255, 0, 0) ,2)
                    #print(faces[0]["box"])
                    face=frame[y:y+h,x:x+w]
                    resize=cv2.resize(face,(50,50))
                    #print(resize.shape)
                    #facep=np.array([1],resize[0],resize[1],resize[2])
                    pred=model.predict_classes(resize.reshape(1,50,50,3))
                    prob=model.predict_proba(resize.reshape(1,50,50,3))
                    print("predictinos=",prob)
                    ind=np.argmax(pred)
                    print(labels)
                    cls=labels.iloc[ind]
                    print(cls)
                    cv2.imshow("faces",face)
                    count-=1
                    if cv2.waitKey(1)==27:
                        break
        cam.release()
        cv2.destroyAllWindows()
        print("dataset created")
predictor()
