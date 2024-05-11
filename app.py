from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from lime import lime_image

from functions.lime_explanation import generate_lime_explanation
from functions.grad_cam import grad_cam
from functions.last_conv import get_last_conv_layer
from functions.plot_results import plot_results
from functions.preprocess_image import preprocess_image


app = Flask(__name__)


@app.route('/')
def upload_file():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    # Load the trained XAI model
    model_path = '/home/deadpool/Desktop/Alzheimer\'s/Works/code X/AD_XAI'
    model = tf.keras.models.load_model(model_path)
    filename =  '/home/deadpool/Desktop/Alzheimer\'s/Works/code X'

    # Define LimeImageExplainer
    explainer = lime_image.LimeImageExplainer()

    # Get the last convolutional layer
    last_conv_layer = get_last_conv_layer(model)

    # Get and Preprocess the uploaded image
    uploaded_file = request.files['file']
    image = Image.open(uploaded_file)
    image = preprocess_image(image)

    # Predict the image
    class_labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    class_index = np.argmax(model.predict(image))
    predicted_class = class_labels[class_index]

    # Generate Lime explanation and Grad-CAM Heat-Map
    lime_explanation = generate_lime_explanation(model, image)
    grad_cam_result = grad_cam(model, image, class_index, last_conv_layer)
    
    plot_results(image, lime_explanation, grad_cam_result, filename, predicted_class, app)
    
    # Render the result template with prediction and explanation
    return render_template('result.html', prediction=predicted_class, plot_images=True)

if __name__ == '__main__':
    app.run(debug=True,port=3000)