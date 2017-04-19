import sudokupy2
import os
import sys
import traceback
from flask import Flask
from flask import render_template, request, make_response
from flask_bootstrap import Bootstrap

import cv2
import numpy as np
import StringIO
from PIL import Image

from base64 import b64encode

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'

curr_image = None

@app.route("/")
def hello():
    global curr_image
    curr_image = None
    return render_template("index.html")

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    global curr_image
    curr_image = None
    if request.method == 'POST':
        # get img and decode for cv2
        try:
            curr_image = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), -1)#cv2.CV_LOAD_IMAGE_UNCHANGED) #changed to -1 for opencv 3
            # encode for cv2 to png
            retval, b = cv2.imencode('.png', curr_image)
            # this allows it to be passed into html directly
            input_img = b64encode(b)
            return render_template("index.html", input_image=input_img)
        except:
            #e = sys.exc_info()
            return render_template("index.html",error="Error decoding image, check that it is a valid image file. ")

      #####input_img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
      #pilImage = Image.open(StringIO(rawImage));
      #npImage = np.array(pilImage)
     # matImage = cv.fromarray(npImage)
    #   return 'file uploaded successfully'
      #done uploading, then run our script and display the image


    #   img = sudokupy2.run(input_img)#app.config['UPLOAD_FOLDER'] + f.filename)
    #   retval, b = cv2.imencode('.png', img)
    #   response = make_response(b.tobytes())
    #   response.headers['Content-Type'] = 'image/png'
      #
    #   output_img = b64encode(b)
      #display output pic




@app.route('/solve', methods = ['GET', 'POST'])
def solve():
    global curr_image
    if request.method == 'POST':
        if curr_image == None:
            return render_template("index.html",error="No image uploaded.")
        try:
            input_img, output_img = sudokupy2.run(curr_image)

            retval, b = cv2.imencode('.png', output_img)
            output_img = b64encode(b)

            retval, b = cv2.imencode('.png', input_img)
            input_img = b64encode(b)
            return render_template("index.html", input_image=input_img, output_image=output_img)
        except:
            # e = sys.exc_info()
            # traceback.print_exc()
            return render_template("index.html",error="Oops! Something went wrong while solving puzzle.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
