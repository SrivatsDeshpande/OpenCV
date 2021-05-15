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

def getObjects(img, draw = True,objects = []):
    classIds, confs, bbox = net.detect(img,confThreshold=thres, nmsThreshold=0.2)
    #print(classIds,bbox)
    if len(objects)==0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([box,className,confidence])
                
    return  img,objectInfo
def tts(objname):
    from gtts import gTTS
    import os
    mytext = 'The object is a ' + objname
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    mytext
    myobj.save('welcome.mp3')
    from pygame import mixer
    mixer.init()
    mixer.music.load('welcome.mp3')
    mixer.music.play()
    
   
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    #cap.set(10,70)
    while True:
        start = time.time()
        success,img = cap.read()
        result, objectInfo = getObjects(img, objects = ['person','bottle','dog','door','chair','mirror','backpack','eye glasses','traffic light','mouse','cell phone','potted plant'])
        #print(objectInfo)
        obj = list(objectInfo)
        if len(obj)>=1:
            objname = obj[0][1]
            acc = obj[0][2]
            tts(objname)
            cv2.waitKey(1500)
            
            
            
        cv2.waitKey(1)
        pass
   
        

