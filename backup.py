
from os import name
import cv2
import time
thres = 0.50 # Threshold to detect object



classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



def unique(names):
    unique_names = []
    for i in names:
        if i not in unique_names:
            unique_names.append(i)
    return unique_names

import pyttsx3
def tts(names): # Function for converting text to speech
    
    
    engine = pyttsx3.init()
    if names == 'Your device is up and running ':
        engine.say('Your device is up and running ')
    elif len(names)>1:
        text = ['The objects are ']
        for i in names:
            i = i.split(',')
            text.append(i)           
        engine.say(str(text))
    elif len(names)==1:
        text = "The object is a "+str(names[0])
        engine.say(text)
    
    
    

    
    engine.runAndWait()
    
    engine.stop()
tts('Your device is up and running ')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)


def getObjects(img,objects = []):
    
    classIds, confs, bbox = net.detect(img,confThreshold=thres, nmsThreshold=0.2)
    
    if len(objects)==0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([box,className,confidence])
              
    return  objectInfo
def capture():
    
    success,img = cap.read()
    
    objectInfo = getObjects(img, objects = ['person','bottle','dog','door','chair','mirror','backpack','eye glasses','traffic light','mouse','cell phone','potted plant'])
    
    return objectInfo
    
def detect_object():
    start = time.time()
    objectInfo = capture()
    
    obj = list(objectInfo)
    names = []
    accuracy = {}
    for i in obj:
        if i[1] not in accuracy.keys():
            accuracy[i[1]] = i[2]
    
    if len(obj)>1:
        
        for i in range(len(obj)):
            names.append(obj[i][1])
        end = time.time()
        
        
    elif len(obj)==1:
        
        names.append(obj[0][1])
        end = time.time()
    else:
        end = time.time()
    names = unique(names)
    
    
    tts(names)
    time.sleep(3)
    return end-start,names,accuracy
    
    
if __name__ == '__main__':       
    while True:
        detect_object()
        
        
    

        
        

        

   
        

