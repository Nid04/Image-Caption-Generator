from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Image

import os
import pickle
import numpy as np
import csv
from django.http import JsonResponse, HttpResponse

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

WORKING_DIR = '/Users/nidhi/Desktop/Image Caption Generation Project/Image-Caption-Generation---SYNCHRONY/kaggle/working/'
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model(os.path.join(WORKING_DIR, 'best_model.h5'))

# still in progress
def feedback(request):
    if request.method == 'POST':
        image_filename = request.POST.get('image_filename')
        feedback_text = request.POST.get('feedback')
        if image_filename and feedback_text:
            # Save feedback to CSV
            with open("user_dataset.csv", mode="a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_filename, feedback_text])

            # Return JSON response for AJAX request
            return JsonResponse({'message': 'Feedback submitted successfully.'})
        else:
            return JsonResponse({'message': 'Invalid form data.'}, status=400)

    return JsonResponse({'message': 'Invalid request.'}, status=400)

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save() 
            # Get the current instance object to display in the template
            img_obj = form.instance
            file_name = img_obj.image
            max_length = 70
            img_feature = extract_features(os.getcwd()+'/media/'+str(file_name))
            caption = predict_caption(model, img_feature, tokenizer, max_length)
            caption = clean_caption(caption)
            return render(request, 'upload/upload.html', {'form': form, 'img_obj': img_obj, 'caption': caption})
    else:
        form = ImageUploadForm()
    return render(request, 'upload/upload.html', {'form': form})


def extract_features(filename):
    # Extract Image Feature
    model = VGG16() 
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # extract feature from image
    image = load_img(filename, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = model.predict(image, verbose = 0)
    return feature


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'start'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequenve
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word is not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'end':
            break
    return in_text

def clean_caption(caption):
    startend_word = ['start', 'end']
    caption = [word for word in caption.split() if word not in startend_word]
    caption = ' '.join(caption)
    return caption