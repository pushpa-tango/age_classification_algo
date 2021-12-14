import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import glob
from skimage.io import imread
from skimage.transform import resize
import os
import time
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import cv2
from statistics import mode
import warnings


ROOT_DIR = os.path.dirname(os.path.abspath('age_test.py'))

time_taken = time.time()
class Age_class:
    def __init__(self):
        print('OOPS')
        self.kid_adult_classification()
        self.kid_age_classification()
        self.adult_age_classification()
    def kid_adult_classification(self):
        filepath = '{}/model/kid_adult_model_arch.txt'.format(ROOT_DIR)
        self.classifier = load_model(filepath)
        model1=self.classifier.load_weights('{}/model/kid_adult_weights.h5'.format(ROOT_DIR))
    def adult_age_classification(self):
        arch_path = '{}/model/adult_model_arch.pb'.format(ROOT_DIR)
        self.adult_classifier = load_model(arch_path)
        model2 = self.adult_classifier.load_weights('{}/model/adult_age_group_classification.h5'.format(ROOT_DIR))
    def kid_age_classification(self):
        arch2_path = '{}/model/kid_model_arch.pb'.format(ROOT_DIR)
        self.kid_classifier = load_model(arch2_path)
        model3 = self.kid_classifier.load_weights('{}/model/kid_age_group_classification.h5'.format(ROOT_DIR))
    def preprocess(self, img):
        norm_img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        pred_img = np.expand_dims(norm_img_resized, axis=0)
        return pred_img
    def age_prediction(self, img, labels, kid_labels, adult_labels):
        pred_img = self.preprocess(img)
        prediction = np.argmax(self.classifier.predict(pred_img), axis=-1)
        if labels[prediction[0]] == 'kid':
            kid_prediction =  np.argmax(self.kid_classifier.predict(pred_img), axis=-1)
            age = kid_labels[kid_prediction[0]]
        elif labels[prediction[0]] == 'adult':
            adult_prediction =  np.argmax(self.adult_classifier.predict(pred_img), axis=-1)
            age = adult_labels[adult_prediction[0]]
        return age

if __name__ == '__main__':
    age_class = Age_class()
    kid_labels = ['age1_12','age13_19']
    adult_labels = ['age20_30','age31_45','age46_60','age60+']
    class_labels = ['adult','kid']
#     age = []
    # Testing dataframes in zips
    path = '{}/test_data/*'.format(ROOT_DIR)
    for image in glob.glob(path):
        img = cv2.imread(image)
        pers_age = age_class.age_prediction(img, class_labels, kid_labels, adult_labels)
        print(pers_age)
#     for zips in glob.glob(path):
#         df = pd.read_pickle(zips)
#         for _,row in df.iterrows():
#             if row['temp_id']<=10000:
#                 age = []
#                 for i in range(len(row['timestamp'])):
#                     if i <= 2:   # Taking only three images of each temp_id
#                         img = np.array(df.iloc[_]['reid_img'][i]).astype('float32') / 255.
#                         age.append(age_class.age_prediction(img, class_labels, kid_labels, adult_labels))
#                 print(mode(age))
print(time.time()-time_taken)
