import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database

# function to get the images and label data
def getImagesAndLabels(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
    folders=[x[0] for x in os.walk('dataset/')]
    names=['None']+[path.split('/')[1] for path in folders[1:]]
    #imagePaths = [os.path.join(path,f) for f in os.listdir(path) if f!='Thumbs.db']  
    for name in names[1:]:
        imagePaths = [os.path.join('dataset/'+name,f) for f in os.listdir('dataset'+'/'+name) if f!='Thumbs.db']  
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        #id = int(os.path.split(imagePath)[-1].split(".")[1])
        id=names.index(imagePath.split('/')[1].split('\\')[0])
        print(id)
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
'''
#print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
'''