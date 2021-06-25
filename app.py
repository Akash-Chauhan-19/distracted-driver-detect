from flask import request
from flask import jsonify
from flask import Flask, render_template
from skimage.feature import local_binary_pattern
import numpy as np
import pickle
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

def get_lbp(images, name='lbp', save=False):
    result = np.array([local_binary_pattern(cv2.cvtColor(images, cv2.COLOR_RGB2GRAY), 10, 3).flatten()])
    if save:
        save_feature(result, name)
    return result


#pca = pickle.load(open('C:\\Users\\MESUT OZIL\\Videos\\BDA Course\\Project\\Machine Learning Projects\\State Farm Distracted Driver Detection\\CACHE\\pca_hog_train_test.pkl','rb'))
#model = pickle.load(open('C:\\Users\\MESUT OZIL\\Videos\\BDA Course\\Project\\Machine Learning Projects\\State Farm Distracted Driver Detection\\CACHE\\knn_pca_model.pkl','rb'))
#min_max_reload = pickle.load(open('C:\\Users\\MESUT OZIL\\Videos\\BDA Course\\Project\\Machine Learning Projects\\State Farm Distracted Driver Detection\\CACHE\\min_max_scaler.pkl','rb'))

pca = pickle.load(open('pca_hog_train_test.pkl', 'rb'))
model = pickle.load(open('knn_pca_model.pkl', 'rb'))
min_max_reload = pickle.load(open('min_max_scaler.pkl', 'rb'))


@app.route("/predict", methods=["POST"])
def predict():
    encoded = request.files['img_form']
    encoded.save(secure_filename(encoded.filename))
    target_size = (64, 64)
    #image = load_img(encoded.filename, target_size=target_size)
    #image_arr = img_to_array(image)

    img = cv2.imread(encoded.filename)
    image = cv2.resize(img, (64, 64))
    image_arr = np.array(image)

    lbp_train = get_lbp(image_arr, name='lbp_train', save=False)
    norm_train = min_max_reload.transform(lbp_train)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # norm_train = min_max_scaler.fit_transform(lbp_train)

    a = pca.transform(norm_train)
    m = model.predict(a)
    os.remove(encoded.filename)
    #K.clear_session()
    response = {'prediction': str(m)}
    return (response)
