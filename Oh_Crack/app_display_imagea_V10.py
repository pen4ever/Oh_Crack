import os
import sys
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory, jsonify, make_response, url_for
from flask_caching import Cache
from flask_table import Table, Col
import io
from io import StringIO
import time
import base64
import string
import random
import json
import numpy as np
from PIL import Image
import keras
import tensorflow
import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from collections import Counter
import gmaps
import gmaps.datasets
import gmaps.geojson_geometries
import json
import geojson
import folium
from branca.colormap import linear
import dash



app = Flask(__name__, template_folder='templates', static_folder='static')
app.cache = Cache(app)  
DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_NAME)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
csvfilename = DIR_NAME + '/Repair_Pavement_Distress_dt_Copy.csv'
filename = ''
PATH_TO_CKPT = DIR_NAME + '/resnet50_csv_lr1e-5_59epoches_inference.h5'
PATH_TO_STATIC = "/".join([APP_ROOT, '/static']) 
NUM_CLASSES = 6
model = None
use_gpu = True
labels_to_names = {0: 'Normal Crack', 1: 'Normal Construction Crack', 2: 'Alligator', 3: 'Pothole'}


# helper functions
def detection_stage(PATH_TO_CKPT, image, scale): 
    # load retinanet model
    model = models.load_model(PATH_TO_CKPT, backbone_name='resnet50')
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    return boxes, scores, labels
    
def HistoryCheck(plist, field, var):
    print(var)
    hist_list = pd.DataFrame(plist[plist[field] == var])
    #print(pd.DataFrame(plist[plist['Street Name'] == 'LOGAN STREET']).head(10))
    #hist_list = pd.DataFrame(plist[plist['Street Name'] =='LOGAN STREET'])
    return hist_list

# Related to the image classification
def id_generator(PATH_TO_STATIC, img_Name_O):
    file_path = os.path.join(PATH_TO_STATIC, img_Name_O)
    filenameO, extension = img_Name_O.split('.', 1)
    timestamp = int(time.time())
    img_Name_N = '{}_{}.{}'.format(filenameO, timestamp, extension)
    return img_Name_N

def LoadMyGeojson(filename):
    with open(filename) as f:
        geometry = json.load(f)
    return geometry

def Pavement_By_Community(Pave_Pro, Pothole_List, Cracked_List, Broken_List):
    Comm_Count = Counter(Pave_Pro['Borough'])
    NumDist = len(list(Comm_Count))
    df2 = pd.DataFrame(columns=['Region','Pothole', 'Cracks', 'BSW', 'Total'])
    for x in list(Comm_Count):
        pnum = len(pd.DataFrame(Pothole_List[Pothole_List['Borough'] == x]))       
        cnum = len(pd.DataFrame(Cracked_List[Cracked_List['Borough'] == x]))
        bnum = len(pd.DataFrame(Broken_List[Broken_List['Borough'] == x]))
        totnum = pnum + cnum + bnum
        df2 = df2.append({'Region': x, 'Pothole': pnum, 'Cracks': cnum, 'BSW': bnum, 'Total':totnum}, ignore_index=True)
    return df2    
    
def FindUniqueWords(data):
    uniquewords = []
    for i in range(len(data)):
        if not data[i] in uniquewords:
            uniquewords.append(data[i])
    return uniquewords 

def CalculateAvgLeadTime(data_list):
    LT = 0
    dtday = []
    count = 0
    for ik in range(len(data_list)):
        s = ''
        s = str(data_list[ik])
        token = s.split()
        dtday.append(int(token[0]))
        LT = LT + (int(token[0]))
        if int(token[0]) == 0:
            count = count + 1
        del token 
    if len(data_list) > 0:
        return int(LT/(len(data_list)-count)), dtday
    else:
        return int(1), dtday
        
img_Name = id_generator(PATH_TO_STATIC,'tmp.png')
pltFileName = "/".join([PATH_TO_STATIC, img_Name])
fields = ['Unique Key', 'Created Date', 'Street Name', 'Cross Street 1','Cross Street 2', 'Community Board', 'Descriptor', 'Status', 'Dt']
fieldMaps = ['Unique Key', 'Created Date', 'Community Board', 'Borough', 'Descriptor', 'Status', 'Dt']

Data_list = pd.read_csv(csvfilename, skipinitialspace=True, usecols=fields)
Data_list.columns = ['Unique Key', 'Created Date', 'Descriptor', 'Street Name', 'Cross Street 1','Cross Street 2', 'Status','Community Board', 'Maintenance Lead Time']
Data_list = Data_list.set_index('Unique Key')

DataMap_list = pd.read_csv(csvfilename, skipinitialspace=True, usecols=fieldMaps)



@app.route("/")
def index():
    return render_template("uploadc.html")

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route("/upload", methods=["POST"])
def upload():
    print('Try to get the form:')
    incident= request.form.get('incident-add')
    street1 = str(request.form['street1'])
    street2 = str(request.form['street2']) # Cross Street 1
    street3 = str(request.form['street3']) # Cross Street 2
    community= str(request.form['community'])
    
    
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for up in request.files.getlist("file"):
        filename = up.filename
        destination = "/".join([target, filename])
        up.save(destination)
         
    print("just filename:", filename )
    
    boxes, scores, labels = predict(destination)  
    print(url_for('static', filename=img_Name))
    print(img_Name)
    print(street1)
    
    # Now try to get the history of the corresponding street
    hist_list = HistoryCheck(Data_list, 'Street Name' , street1.upper())
    
    print('list')
    print(len(hist_list))
    if len(hist_list) == 0: 
        # estimate based on community 
        comm_list = HistoryCheck(Data_list, 'Community Board' , community)
        leadavg, dtday = CalculateAvgLeadTime(comm_list['Maintenance Lead Time'].values.tolist())
        comm_list['Maintenance Lead Time'] = dtday
        return render_template("complete_display_imagede.html", image_name=img_Name, crack_type=labels, lead_time = leadavg, tables=[comm_list.to_html(classes='data')],titles=comm_list.columns.values)
    else:
        leadavg, dtday = CalculateAvgLeadTime(hist_list['Maintenance Lead Time'].values.tolist())
        hist_list['Maintenance Lead Time'] = dtday
        hist1_list = hist_list[hist_list['Maintenance Lead Time'] >0]
        return render_template("complete_display_imagede.html", image_name=img_Name, crack_type= labels, lead_time = leadavg, tables=[hist1_list.to_html(classes='data')], titles=hist1_list.columns.values)

    #return render_template("complete_display_imaged.html", image_name="../static/"+img_Name, crack_type=img_Name)

@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory("static", filename)

@app.route('/upload.png')
def plot_png(image_np,boxes, scores, classes, num,category_index, data):
    vis_util.visualize_boxes_and_labels_on_image_array(
     image_np,
     np.squeeze(boxes),
     np.squeeze(classes).astype(np.int32),
     np.squeeze(scores),
     category_index,
     min_score_thresh=0.3,
     use_normalized_coordinates=True,
     line_thickness=8)  
    fig = plt.gcf()
    plt.imshow(image_np)
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    response=make_response(output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    plot_url = base64.b64encode(output.getvalue())

    return plot_url

@app.route("/upload", methods=["POST"])
def predict(filename):
    
    # Initialize the data dictionary 
    data = {"success": False}
    print("prediction block before request")
    # Ensure an image was properly uploaded to our endpoint.
    if request.method == 'POST':
         
         image = cv2.imread(filename)
       
         draw = image.copy()
         draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        
         image = preprocess_image(image)
         image, scale = resize_image(image, 300, 300)
         
    
         boxes, scores, labels = detection_stage(PATH_TO_CKPT, image, scale)
         boxes /= scale
         labelstr = []
         for box, score, label in zip(boxes[0], scores[0], labels[0]):             
            # scores are sorted so we can break
             if label >= 0:               
                print('score', score)
                print('label', labels_to_names[label])
                if score > 0.5:
                    labelstr.append(labels_to_names[label])
                    color = label_color(label)               
                    b = box.astype(int)
                    draw_box(draw, b, color=color)          
                    caption = "{}".format(labels_to_names[label])
                #draw_caption(draw, b, caption)
         #cv2.imshow("Show",draw)
         cv2.imwrite(pltFileName, draw)
         
         uniquelabel = FindUniqueWords(labelstr)
         print(len(uniquelabel))
         Crack = ''
         for kkk in range(len(uniquelabel)):
             if kkk != len(uniquelabel)-1:
                  Crack = Crack + str(uniquelabel[kkk]) + ','
             else:
                  Crack = Crack + str(uniquelabel[kkk])
          
    return  boxes, scores, Crack

    
if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.run(host='0.0.0.0', debug=True,use_reloader=True)

