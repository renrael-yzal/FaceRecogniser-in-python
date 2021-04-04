import os
from mtcnn.mtcnn import MTCNN
import cv2
#from matplotlib.patches import Rectangle
import numpy as np
def face_extractor(count,name):
        name=name.title()
        l=os.listdir("dataset")
        if name in l:
                print("you have same person in data do you want to continue(y\n):-")
                if n:
                        return
                else:
                        face_extractor(count,name)
        else:
                os.mkdir("dataset\\"+name)
        face_detector=MTCNN()
        cam=cv2.VideoCapture(0)
        while count>0:
                ret,frame=cam.read()
                faces=face_detector.detect_faces(frame)
                if len(faces)>0:
                    x,y,w,h=faces[0]["box"]
                    #frame=cv2.rectangle(frame,(w,h),(x,y),(255, 0, 0) ,2)
                    cv2.imwrite("dataset\\"+name+"\\"+name+str(count)+".jpg",frame[y:y+h,x:x+w])
                    #print(faces[0]["box"])
                    face=frame[y:y+h,x:x+w]
                    resize=cv2.resize(face,(50,50))
                    cv2.imshow("faces",face)
                    count-=1
                    if cv2.waitKey(1)==27:
                        break
        cam.release()
        cv2.destroyAllWindows()
        print("dataset created")
        return
#face_extractor(50,"mani")
no=150#int(input("enter no of samples:="))
while True:
        name=input("enter name of candidate or -1 to stop:=")
        if name=="-1":
                break
        try:
                face_extractor(no,name)
        except Exception as e:
                print(e)
                print("place the face facing to camera")
        
        
        
        














"""import cv2
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Frame
from PIL import Image, ImageTk

white 		= "#ffffff"
lightBlue2 	= "#adc5ed"
font 		= "Constantia"
fontButtons = (font, 12)
maxWidth  	= 600
maxHeight 	= 600

#Graphics window
mainWindow = tk.Tk()
mainWindow.configure(bg=lightBlue2)
mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
#mainWindow.resizable(0,0)
# mainWindow.overrideredirect(1)

mainFrame = Frame(mainWindow)
mainFrame.place(x=0, y=20)                

#Capture video frames
lmain = tk.Label(mainFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture(0)

def show_frame():
	ret, frame = cap.read()

	cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	img   = Image.fromarray(cv2image).resize((760, 400))
	imgtk = ImageTk.PhotoImage(image = img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(10, show_frame)

label=tk.Label(text="dataset length")
label.place(x=100,y=430)
inp=tk.Text(mainWindow,width=5,height=1)
inp.place(x=200,y=430)
closeButton = tk.Button(mainWindow, text = "CLOSE", bg = white, width = 5, height= 1)
closeButton.configure(command= lambda: mainWindow.destroy())              
closeButton.place(x=0,y=430)	

show_frame()  #Display
mainWindow.mainloop()  #Starts GUI

"""
