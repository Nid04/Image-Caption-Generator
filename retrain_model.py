import os
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.config.run_functions_eagerly(False)


BASE_DIR = os.getcwd()

# extract features
def extract_features():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # extract feature from image
    features = {}
    directory = os.path.join(BASE_DIR, 'mysite/media/images')

    for img_name in os.listdir(directory):
        img_path = directory + '/' + img_name
        image = load_img(img_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for vgg
        image = preprocess_input(image)
        # extract features
        feature = model.predict(image, verbose = 0)
        # get image ID
        image_id = img_name.split('.')[0]
        # store feature
        features[image_id] = feature
    return features

# load the document
def load_caption():
    with open(os.path.join(BASE_DIR, 'mysite/user_dataset.csv'), 'r') as f:
        captions_doc = f.read()

    # create mapping of image to captions
    mapping = {}
    # process lines
    for line in captions_doc.split('\n'):
        # split the line by comma(,)
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        # remove extensions from image ID
        image_id = image_id.split('.')[0]
        # convert caption list to string
        caption = " ".join(caption)
        # create list if needed
        if image_id not in mapping:
            mapping[image_id] = []
        #store the caption
        mapping[image_id].append(caption)
    return mapping

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # replace digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'start ' + " ".join([word for word in caption.split() if len(word)>1]) +' end'
            captions[i] = caption 

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    x1, x2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode the output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store the sequences
                    x1.append(features[key][0])
                    x2.append(in_seq)
                    y.append(out_seq)
                    
            if n == batch_size:
                x1, x2, y = np.array(x1), np.array(x2), np.array(y)
                yield [x1, x2], y
                x1, x2, y = list(), list(), list()
                n = 0

features = extract_features()
mapping = load_caption()
clean(mapping)

# storing all the captions
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# getting the max length
max_length = max(70, max(len(caption.split()) for caption in all_captions))

# loading the existing tokenizer
with open(BASE_DIR + "/kaggle/working/tokenizer.pkl", 'rb') as t:
    tokenizer = pickle.load(t)

tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index)+1

with open(BASE_DIR + "/kaggle/working/tokenizer1.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# loading pretrained model
pretrained_model_path = BASE_DIR + "/kaggle/working/best_model.h5"
pretrained_model = load_model(pretrained_model_path)

new_learning_rate = 1e-5 # to add less weight to the trained model
optimizer = Adam(learning_rate=new_learning_rate)
pretrained_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 64
train = list(mapping.keys())
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    pretrained_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

pretrained_model.save(BASE_DIR + '/kaggle/working/retrain_model1.h5')
