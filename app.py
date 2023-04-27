import math 
import os
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import cv2 as cv2
import imutils
#from matplotlib import pyplot as plt
import dlib
#import tensorflow as tf
import sys
import datetime
from PIL import Image
#from PIL import ImageFont
#from PIL import ImageDraw
from rembg.bg import remove
from imutils import face_utils
from cairosvg import svg2png
import io
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import json

def drawSegment(img):
  
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	# load image
  
  #os.remove(outputFilePath+'/'+fileop[0]+'.png') 
  h, w, ch = img.shape
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# add an alpha channel to image
  b,g,r = cv2.split(img)
  a = np.ones((h,w,1), np.uint8) * 255
  img = cv2.merge((b, g, r, a))
	# detect face
  rects = detector(gray,1)
  
  if 0 == len(rects):
    return None
	
  roi = rects[0] # region of interest
  shape = predictor(gray, roi)
  shape = face_utils.shape_to_np(shape)
  
  jawline = shape[4:14]
  top = min(jawline[:,1])
  bottom = max(jawline[:,1])
	# extend contour for masking
  jawline = np.append(jawline, [ w-1, jawline[-1][1] ]).reshape(-1, 2)
  jawline = np.append(jawline, [ w-1, h-1 ]).reshape(-1, 2)
  jawline = np.append(jawline, [ 0, h-1 ]).reshape(-1, 2)
  jawline = np.append(jawline, [ 0, jawline[0][1] ]).reshape(-1, 2)
  contours = [ jawline ]
	# generate mask
  mask = np.ones((h,w,1), np.uint8) * 255 # times 255 to make mask 'showable'
  cv2.drawContours(mask, contours, -1, 0, -1) # remove below jawline
	# apply to image
  result = cv2.bitwise_and(img, img, mask = mask)
	#result = cv2.add(img,img, mask = mask)

  src = result
  tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
  b, g, r, m = cv2.split(src)
  rgba = [b,g,r, alpha]
  dst = cv2.merge(rgba,4)

	# extract jawline
  x_val = shape[27][0] - shape[0][0]
  y_val = shape[8][1] - shape[27][1]
  
  ratio_val = x_val/y_val
  
  ratio_val = math.sqrt(12755 * 4/(3.14156 * ratio_val))/y_val
  
  dst = cv2.resize(dst, (int(dst.shape[1] *  ratio_val), int(dst.shape[0]*  ratio_val)), interpolation = cv2.INTER_AREA)
  img = cv2.resize(img, (int(img.shape[1] *  ratio_val), int(img.shape[0]*  ratio_val)), interpolation = cv2.INTER_AREA)
  
  y_diff_1 = int(dst.shape[0] - (shape[8][1] * ratio_val))
  y_diff_2 = int((shape[8][1]  * ratio_val) - 0)
  x_diff_1 = int(dst.shape[1] - (shape[8][0]  * ratio_val))
  x_diff_2 = int((shape[8][0]  * ratio_val) - 0)
  
  if(y_diff_1 < 56):
    layer = np.zeros((56 - y_diff_1, dst.shape[1], 4))
    dst = np.concatenate ((dst, layer), axis=0)
    img = np.concatenate ((img, layer), axis=0)
  else:
    dst =  dst[0:56 +  int(shape[8][1] * ratio_val),:,:]
    img =  img[0:56 +  int(shape[8][1] * ratio_val),:,:]

  if(y_diff_2 < 457):
    layer = np.zeros((457 - y_diff_2 ,dst.shape[1],  4))
    dst = np.concatenate ((layer,dst), axis=0)
    img = np.concatenate ((layer,img), axis=0)
  else:
    dst = dst[dst.shape[0] - (513):dst.shape[0] ,:,:]
    img = img[img.shape[0] - (513):img.shape[0] ,:,:]
	
  if(x_diff_1 < 251):
    layer = np.zeros((dst.shape[0] , 251 - x_diff_1, 4))
    dst = np.concatenate ((dst, layer), axis=1)
    img = np.concatenate ((img, layer), axis=1)
  else:
    dst =  dst[:,0:251 +  int(shape[8][0] * ratio_val),:]
    img =  img[:,0:251 +  int(shape[8][0] * ratio_val),:]
  
  if(x_diff_2 < 262):
    layer = np.zeros((  dst.shape[0] , 262 - x_diff_2, 4))
    dst = np.concatenate ((layer,dst), axis=1)
    img = np.concatenate ((layer,img), axis=1)
  else:
    dst = dst[:,dst.shape[1] - 513:dst.shape[1] ,:]
    img = img[:,img.shape[1] - 513:img.shape[1] ,:]

  img2 = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)

  img = cv2.resize(dst, (300, 300), interpolation = cv2.INTER_AREA)

  img = remove(cv2.imencode('.jpg', np.uint8(img))[1].tobytes())
  
  #img = Image.open(io.BytesIO(img)).convert("RGBA")
  #print(list(io.BytesIO(img).read()))
  
  #img.save("b.png")

  return [list(io.BytesIO(img).read()), list(cv2.imencode('.png', img2)[1].tostring())]

app = Flask(__name__)
CORS(app)
@app.route("/")
def hello_world():
    return jsonify({'status': 404})

@app.route('/makefile', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        intArr = json.loads( request.data )
        img_encoded = bytes(intArr)
        nparr = np.fromstring(img_encoded, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        print(request.args.get('mimeType'))
        #if request.mimeType
        try:
          img = drawSegment(img_np)
          
          if img != None:
            return jsonify({'status': 100, 'cropped':img[0], 'resized':img[1]})
          else:
            return jsonify({'status': 200})
        except Exception as e:
          print(e)
          return jsonify({'status': 300})
        #img = Image.open(io.BytesIO(img_encoded)).convert("RGBA")
    else:
        return jsonify({'status': 404})


@app.route('/makesvg', methods=['POST'])
def svgToPng ():
  intArr = json.loads( request.data )
  byteArr = []

  try:
    for svg in intArr:
      byteArr.append(svg2png(bytestring=svg))
    return jsonify({'status':100,'data':byteArr})
  except Exception as e:
    return jsonify({'status':200,'data': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))