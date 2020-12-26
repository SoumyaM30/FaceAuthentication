from flask import Flask, url_for
import json
from flask_cors import CORS
from flask import request
from flask import jsonify
from werkzeug.utils import secure_filename
from binascii import a2b_base64
from base64 import b64decode
from base64 import decodestring
#from w3lib.url import parse_data_uri
import cv2
import numpy as np
import os 
from face_detect import capture_training_examples
from face_trainer import getImagesAndLabels
from face_recognizer import identify

app = Flask(__name__)
cors = CORS(app)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        folders=[x[0] for x in os.walk('dataset/')]
        names=['None']+[path.split('/')[1] for path in folders[1:]]
        name=identify()
        return jsonify(name)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        face_name=request.form['name']
        if not os.path.isdir("dataset/"+face_name):
            os.makedirs("dataset/"+face_name)
        capture_training_examples(face_name)

        faces,ids = getImagesAndLabels("dataset/")
        print(ids)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        recognizer.train(faces, np.array(ids))
        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')
        return jsonify("user registered")


if __name__ == '__main__':
    print("Listening...")
    app.run(host='0.0.0.0')