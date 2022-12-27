# libraries
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from numpy.linalg import norm
import requests
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTImageProcessor
import os

# no mostrar los warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ##### Models ##########
# VIT Model
preprocess_img = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Resnet50v2-avg Pooling
model_resnet50_v2_avg = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',  # 'imagenet' (pre-training on ImageNet).
    input_tensor=None,
    input_shape=None,
    pooling='avg',  # global avg pooling will be applied
)


# ########### Functions ######################


def crop_product(url_img, blur_kernel=15):
    '''crops a product from an image'''
    try:
        img = Image.open(requests.get(url_img, stream=True).raw)
        img_mode = img.mode

    except Exception as e:
        return e

    # change channels to RGB
    if img_mode == 'RGBA':
        img_aux = Image.new("RGB", img.size, (255, 255, 255))
        img_aux.paste(img, mask=img.split()[3])
        img = img_aux

    if img_mode == 'CMYK':
        img = img.convert('RGB')

    if img_mode == 'P':
        img = img.convert('RGB', palette=Image.ADAPTIVE, colors=256)

    img = np.array(img)
    #thresholding
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    gray = cv2.blur(gray, (blur_kernel, blur_kernel))
    thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY_INV)[1]
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1] # no funciona bien..como determina automatico el threshold confunde los colores claros con el fondo blanco
                                                                                #BUENO para el caso en que el fondo no es blanco y el color es muy distinto al del objeto
    alpha = 1  # for undefined cases : x/0 (no white pixels)
    ratio = cv2.countNonZero(thresh)/((img.shape[0] * img.shape[1]) - cv2.countNonZero(thresh) + alpha)

    if ratio > 2:  # no crop. ratio=2 good enough?
        cropped = img
        return img, cropped, thresh, ratio

    # ratio<2,  getting the max countour from img (product)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    max_a = 0
    for contour in contours:
        x_aux, y_aux, w_aux, h_aux = cv2.boundingRect(contour)
        a = w_aux * h_aux
        if a > max_a:
            max_a = a
            x, y, w, h = x_aux, y_aux, w_aux, h_aux

    cropped = img.copy()[y:y + h, x:x + w]
    return img, cropped, thresh, ratio


def cosine_distance(url_img1, url_img2, model, crop=0):
    '''calculates cosine distance between 2 images'''
    assert (model == model_vit or model == model_resnet50_v2_avg), 'wrong input for model' #Sólo estos 2 modelos disponibles
    assert crop in {0, 1}, 'no crop:0, crop:1'
    try:
        if crop:
            img1 = url_img1
            img2 = url_img2

        if not crop:  # change channels to RGB

            img1 = Image.open(requests.get(url_img1, stream=True).raw)
            img2 = Image.open(requests.get(url_img2, stream=True).raw)

            imgs = []
            for img in [img1, img2]:

                if img.mode == 'RGBA':  # RGBA
                    img_aux = Image.new("RGB", img.size, (255, 255, 255))
                    img_aux.paste(img, mask=img.split()[3])
                    img = img_aux

                if img.mode == 'CMYK':
                    img = img.convert('RGB')

                if img.mode == 'P':
                    img = img.convert('RGB', palette=Image.ADAPTIVE, colors=256)

                imgs.append(img)
            img1 = imgs[0]
            img2 = imgs[1]

        #Generating the embedding for each img depending on the model used.
        embeddings = []
        for img in [img1, img2]:
            if model == model_vit:
                #preprocessing
                inputs = preprocess_img(img, return_tensors="pt")
                #embedding
                embedding = model(**inputs).last_hidden_state[0][0].detach().numpy()
                embeddings.append(embedding)

            if model == model_resnet50_v2_avg:
                #preprocessing
                img = np.array(img)
                img = tf.keras.applications.resnet_v2.preprocess_input(img)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(img, axis=0)
                #embedding
                embedding = model.predict(img)[0]
                embeddings.append(embedding)

        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        #cosine_distance
        distance = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

    except Exception as e:
        print(e)
        return -1

    if distance > 1:
        return 1
    if distance < 0:
        return 0

    return distance


def thresholding_display(img1, img2):
    '''plots the base image, thresholded and cropped'''

    img1_display,img1_cropped, img1_threshold, img1_ratio = crop_product(img1)
    img1_ratio = round(img1_ratio, 3)

    img2_display,img2_cropped, img2_threshold, img2_ratio = crop_product(img2)
    img2_ratio = round(img2_ratio, 3)

    # define subplots
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
    #Customer
    #original
    ax[0,0].set_xlabel('pixels')
    ax[0,0].set_ylabel('pixels')
    ax[0,0].set_title("Original Cliente, crop=0")
    ax[0,0].imshow(img1_display)
    #thresholded
    ax[0,1].set_title("Thresholded")
    ax[0,1].set_xlabel(f'Ratio: {img1_ratio}')
    ax[0,1].imshow(img1_threshold)
    #cropped
    ax[0,2].set_title("Cropped, crop=1")
    ax[0,2].imshow(img1_cropped)

    #Retail
    ax[1,0].set_xlabel('pixels')
    ax[1,0].set_ylabel('pixels')
    ax[1,0].set_title("Original Retail, crop=0")
    ax[1,0].imshow(img2_display)

    ax[1, 1].set_title("Thresholded")
    ax[1, 1].set_xlabel(f'Ratio: {img2_ratio}')
    ax[1, 1].imshow(img2_threshold)

    ax[1,2].set_title("Cropped, crop=1")
    ax[1,2].imshow(img2_cropped)

    fig.tight_layout()
    return fig


def similarity_score(url_img1, url_img2, model, crop=1, blur_kernel=15):
    '''calculates the similarity score between 2 imgs using a defined model'''
    if crop:
        img_cliente = crop_product(url_img1, blur_kernel=blur_kernel)[1]
        img_retail = crop_product(url_img2, blur_kernel=blur_kernel)[1]
        score = cosine_distance(img_cliente, img_retail, model, crop=crop)

    if not crop:
        score = cosine_distance(url_img1, url_img2, model, crop=crop)

    return score

def check_url(url):
    ''''returns code status from requesting a url'''
    code = requests.head(url).status_code
    return code
