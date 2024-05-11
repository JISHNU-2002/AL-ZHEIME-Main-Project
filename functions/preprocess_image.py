import numpy as np

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((176, 208))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image
