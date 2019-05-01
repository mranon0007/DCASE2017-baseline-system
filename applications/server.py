from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "uploads" 

def runTask(filename, task = 1):
    if task == 1:
        # import task1Api
        # return task1Api.run("a016_120_130.wav")
        return '''{"/home/yusufkhanjee/FYP/applications/data/uploads/a016_120_130.wav": "beach"}'''

    elif task == 2:
        # import task2Api
        # return task2Api.run("mixture_devtest_GUNSHOT_478_7cddfc5abf0fe86d4c2de430da87a7c3.wav")
        return ''' {"glassbreak": [["17.4", "18.0"]], "gunshot": [["17.4", "18.599999999999998"]], "babycry": [["17.4", "18.0"]]}'''
    
    else: 
        # error
        pass

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/uploader")
def a():
    return render_template('test.html')

@app.route("/upload", methods = ['POST'])
def hello():
    # request.form['key']
    files = request.files
    
    if "classifying":
        # run task 1
        # return results
        pass
    elif "detecting":
        # run task 2
        # return results
        pass
    else:
        #throw error
        pass

    # No file uploaded
    if 'audioFile' not in request.files:
        return 'No file part'

    else:
        #if file uploaded
        audioFile = request.files['audioFile']
        audioFile.save(os.path.join(os.path.dirname(__file__), "data/uploads", secure_filename(audioFile.filename)))
        return secure_filename(audioFile.filename)

    return "Hello World!"

    res = runTask("a", task = 2)
    return res

app.run(debug=True, host= '0.0.0.0', port=4996)
