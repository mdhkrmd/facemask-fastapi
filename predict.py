from PIL import Image
from keras.applications.mobilenet import preprocess_input
import numpy as np
import keras
from keras.models import load_model

def proses(file):
    model_baru=load_model('model.h5')
    jenis = ['incorrect_mask', 'with_mask', 'without_mask']

    image = Image.open(file.file)
    image = image.convert('RGB')
    image = image.resize((150,150))
    image = np.asarray(image)
    image = np.expand_dims(image,0)
    image = image/255.0
    # gambar = keras.applications.mobilenet.preprocess_input(image)
    
    p = model_baru.predict(image)
    kelas = p.argmax(axis = 1)[0]
    label = jenis[kelas]
    conf = p[0][kelas]
    return conf, label