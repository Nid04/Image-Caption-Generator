import os
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


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

# Function to selectively transfer weights between models
def transfer_weights(source_model, target_model):
    target_layers = {layer.name: layer for layer in target_model.layers}
    for layer in source_model.layers:
        if layer.name in target_layers and layer.get_weights():
            if layer.get_weights()[0].shape == target_layers[layer.name].get_weights()[0].shape:
                target_layers[layer.name].set_weights(layer.get_weights())

def define_model():
    # loading pretrained model
    old_model_path = BASE_DIR + "/kaggle/working/best_model.h5"
    old_model = load_model(old_model_path)
    #plot_model(pretrained_model, show_shapes=True)
    # Create a new model with updated max_length and vocab_size
    # Image feature model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    new_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    new_model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Load the pre-trained weights into the new model
    transfer_weights(old_model, new_model)
    return new_model

def train_model():
    epochs = 20
    batch_size = 3
    image_ids = list(mapping.keys())
    train = image_ids
    steps = len(train) // batch_size

    for i in range(epochs):
        # create data generator
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        # fit for one epoch
        new_model = define_model()
        new_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    new_model.save(BASE_DIR + '/kaggle/working/retrain_model1.h5')

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

train_model()