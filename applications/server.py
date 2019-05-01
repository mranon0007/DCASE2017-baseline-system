from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "uploads" 

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

    import task1Api
    return task1Api.run("a016_120_130.wav")
    # {"/home/yusufkhanjee/FYP/applications/data/uploads/a016_120_130.wav": "beach"}

    import task2Api
    return task2Api.run("mixture_devtest_GUNSHOT_478_7cddfc5abf0fe86d4c2de430da87a7c3.wav")
    # {"glassbreak": [["17.4", "18.0"]], "gunshot": [["17.4", "18.599999999999998"]], "babycry": [["17.4", "18.0"]]}
    
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


    # # No file uploaded
    # if 'audioFile' not in request.files:
    #     return 'No file part'

    # else:
    #     #if file uploaded
    #     audioFile = request.files['audioFile']
    #     audioFile.save(secure_filename(audioFile.filename))
    #     return secure_filename(audioFile.filename)

    return "Hello World!"

app.run(debug=True, host= '0.0.0.0')