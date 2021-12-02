from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
import cv2


app = Flask(__name__)

model = tf.keras.models.load_model('model_19.h5')


################################################################################################################################################
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image


def generate_desc(model, tokenizer, photo, max_len):
    
    '''
        This function generate_desc() implements this behavior and generates a textual description given a trained model,
        and a given prepared photo as input. 
        It calls the function word_for_id() in order to map an integer prediction back to a word.
    '''

    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_len):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_len)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text




# load the tokenizer
tokenizer = pickle.load(open(r'tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 40


# extract features from each photo in the directory
def extract_features_one(img):
    # load the model
    model = tf.keras.applications.vgg16.VGG16()
    # re-structure the model
    model = tf.keras.models.Model(
        inputs=model.inputs, outputs=model.layers[-2].output)
    # prepare the image for the VGG model
    image = tf.keras.applications.vgg16.preprocess_input(img)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


#####################################################################################################################################

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():

    # global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")

    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = np.reshape(image, (1, 224, 224, 3))

    # incept = resnet.predict(image).reshape(1, 2048)

    print("="*50)
    print("Predict Features")

    photo = extract_features_one(image)
    # print(photo.shape)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)

    description = description.split(' ',1)[1]
    description = description.rsplit(' ', 1)[0]

    return render_template('after.html', data=description)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)