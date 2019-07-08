'''Flask Server to run Satellite Image Segmentation '''
__author__ = "Adithya Sampath"
__date__  = "July 2019"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
import os
from torch.autograd import Variable
import cv2
#image utils
from skimage.transform import rescale
from skimage.io import imread, imsave
import numpy as np
#FLASK REST utils
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
#misc
import os
import sys
import time
#segnet
from SegNet import *
#init flask
app = Flask(
    __name__,
    static_url_path="/images",
    static_folder="images",
    template_folder='templates')
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
valid_mimetypes = ['image/jpeg', 'image/png', 'image/tiff']

net = SegNet()
net.cuda()
net.load_state_dict(
    torch.load(
        os.path.join(APP_ROOT, "weights", 'segnet_final_reference.pth')))


def predict(img_path, img_name, net):
    preds = test(net, img_path, all=True, stride=32)
    img = convert_to_color(preds)

    img_name_disp = 'inference_{}'.format(img_name)
    pred_img = os.path.join(".", "images", img_name_disp)
    io.imsave(pred_img, img)
    return pred_img, img_name_disp


def find_building(image_path, image_name):
    img = np.asarray(cv2.imread(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    image[np.where((image == [0, 0, 255]).all(axis=2))] = [255, 0, 255]
    image[np.where((image == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    image[np.where((image == [0, 255, 255]).all(axis=2))] = [0, 0, 0]
    image[np.where((image == [0, 255, 0]).all(axis=2))] = [0, 0, 0]
    image[np.where((image == [255, 255, 0]).all(axis=2))] = [0, 0, 0]
    image[np.where((image == [255, 0, 0]).all(axis=2))] = [0, 0, 0]
    img_name_disp = "building_" + str(image_name).split(".")[0] + ".png"
    img_path_disp = os.path.join(".", "images", img_name_disp)
    imsave(img_path_disp, image)
    return img_path_disp


#flask functions
@app.route('/')
def index():
    return render_template("upload.html")


@app.route('/getImage', methods=["POST"])
def upload():
    global net
    image_folder = os.path.join(APP_ROOT, "images")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400

        #for inference
        img_file_pred = request.files['file']
        image_name_pred = img_file_pred.filename
        mim_type = img_file_pred.content_type
        if not mim_type in valid_mimetypes:
            return jsonify({'error': 'bad file type'}), 400
        img_path_pred = os.path.join(".", "images", image_name_pred)
        img_file_pred.save(img_path_pred)
        print(image_name_pred)

        #for browser display
        img_file_disp = np.asarray(
            rescale(io.imread(img_path_pred), 1.0 / 4.0, anti_aliasing=False),
            dtype='float32')
        img_name_disp = str(img_file_pred.filename).split(".")[0] + ".png"
        img_path_disp = os.path.join(".", "images", img_name_disp)
        imsave(img_path_disp, img_file_disp)
        print(img_name_disp)
        start = time.time()
        pred_img, pred_name = predict(img_path_pred, img_name_disp, net)
        building_img = find_building(pred_img, pred_name)
        print("Prediction time: ", time.time() - start)
        return render_template(
            "complete.html",
            image_name=img_path_disp,
            out_name=pred_img,
            building_name=building_img)


if __name__ == "__main__":
    PORT = 5000
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    app.run(host='0.0.0.0', port=PORT)