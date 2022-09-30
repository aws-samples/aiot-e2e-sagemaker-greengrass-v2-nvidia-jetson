import logging, sys
import cv2
import glob
import json
import numpy as np
import dlr
from dlr import DLRModel


def load_classes_dict(filename='classes_dict.json'):
    with open(filename, 'r') as fp:
        classes_dict = json.load(fp)

    classes_dict = {int(k):v for k,v in classes_dict.items()}        
    return classes_dict
    

def load_image(image_path):
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    return image_data


def preprocess_image(image, image_shape=(224,224)):
    cvimage = cv2.resize(image, image_shape)
    img = np.asarray(cvimage, dtype='float32')
    img /= 255.0 # scale 0 to 1
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1)) 
    img = np.expand_dims(img, axis=0) # e.g., [1x3x224x224]
    return img


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    f_x = x_exp / np.sum(x_exp)
    return f_x


device = 'cpu'
model = DLRModel(f'model_{device}', device)
sample_image_dir = 'sample_images'
classes_dict = load_classes_dict('classes_dict.json')

extensions = (f"{sample_image_dir}/*.jpg", f"{sample_image_dir}/*.jpeg")
img_filelist = [f for f_ in [glob.glob(e) for e in extensions] for f in f_]
print(img_filelist)

for img_filepath in img_filelist:
    ground_truth = img_filepath.split('/')[-1]
    img = load_image(img_filepath)
    img_data = preprocess_image(img)
    
    output = model.run(img_data)  
    probs = softmax(output[0][0])
    sort_classes_by_probs = np.argsort(probs)[::-1]

    idx = sort_classes_by_probs[0]
    print("+"*80)
    print(f'predicted = {classes_dict[idx]}, {probs[idx]*100:.2f}%')
    print(f'ground_truth = {ground_truth}')  
    

