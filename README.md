# Enhancing Early Diagnosis of Alzheimer's Disease Through Machine Learning

A brief description of what this project does and who it's for

This repository contains the implementation of the Machine Learning project aimed at revolutionizing the early diagnosis of Alzheimer's Disease (AD) using machine learning techniques, particularly Convolutional Neural Networks (CNNs), and explainable AI.

## 01 Introduction

Alzheimer's Disease is the leading cause of dementia, affecting millions globally. Early detection is crucial for timely intervention, but current diagnostic methods have limitations. This project aims to enhance early diagnosis using deep learning models that analyze MRI data and provide interpretable predictions through explainable AI techniques.

## 02 Project Objectives
- **Prediction** : To predict the stage of Alzheimer's disease from brain MRI images using CNNs
- **Interpretability** : To understand and interpret the predictions using explainable AI methods

## 03 System Development & Architecture

The system architecture uses a CNN model to analyze brain MRI data and includes components for data preprocessing, model training, evaluation, and deployment of explainable AI techniques.

### Dataset Collection & Preprocessing
- Brain MRI data was sourced from public datasets such as Kaggle
- The dataset contains total of 6400 records. It has image data classified into 4 classes
- 0 `Mild Demented`, 1 `Moderate Demented`, 2 `Non-Demented`, 3 `Very Mild Demented`

```bash
https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset
```

- In Preprocessing stage, some transformations are applied on the input image

```Python
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((176, 208))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image
```

### Model Training and Evaluation

- A specialized CNN model is developed to capture subtle patterns indicative of Alzheimer's
- The Dataset is divided into two parts, with the training data containing 80% of the images in each label, and the test data containing 20% of the images in each label

### CNN Model 

![CNN Model](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/model.png)

#### (1) Convolution layer 
- This layer performs the convolution operation with a kernel size of 3 x 3
- The output of the convolution operation is a set of feature map where each of the feature map represents the the activation of different positions in the input image
- The output feature map size is 208 x 176 x 32

#### (2) MaxPooling layer
- Poolig window size 2 x 2
- MaxPooling helps in reducing spatial dimensions of our image while retaining the relevant information
- The reduced feature map size after pooling operation is 104 x 88 x 32

#### (3) Convolution layer
- Window size 3 x 3
- Output feature map of size 104 x 88 x 32

#### (4) MaxPooling layer
- Window size 2 x 2
- Output feature map of size 52 x 44 x 32

#### (5) Dropout layer 
- Added to reduce overfitting
- 0.5 = 50 %, so throughout the training process drop 1/2 of what the resultant to keep the model learning new approaches and not become stale

#### (6) Flatten layer 
- To flatten the output of the convolutional layer to one dimensional vector

#### (7) Output layer
- Dense layer consisting of 1 neuron
- The activation function used in the first dense layer is ReLU activation function
- The final dense layer outputs soft probabilities for the mild demented, moderate demented, non-demented, very mild demented classes using `softmax` activation function

### Model Parameters
**Activation function**

(1) **`ReLU ( Rectified Linear Unit )`**
- Most commonly used activation function turns negative values into 0 and outputs positive values 1
- An Activation function is responsible for taking inputs and assigning weights to output nodes, a model can’t coordinate itself well with negative values, it becomes non-uniform in the sense of how we expect the model to perform, `f(x) = max(0, x)` 
- The output of a ReLU unit is non-negative
- Returning x, if less than zero then max returns 0

(2) **`Softmax function`**
- Used in the output layer for multi-class classification to convert raw scores into probabilities
- Normalizes input values into a probability distribution where the sum of all probabilities equals 1
- `Softmax(zi​) = ezi​ / ∑j=1K​ ezj`​
- Making the highest score stand out, which is useful for clear probabilistic predictions in multi-class classification problems

**Optimization function**
- `Adam optimization` is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments

**Loss Function**
- The purpose of loss functions is to compute the quantity that a model should seek to minimize during training
- `sparse categorical crossentropy` is used as a loss function when the output labels are represented in a sparse matrix format

**Evaluation Metrics**
- To evaluate the performance or quality of the model performance metrics or evaluation metrics such as `accuracy,precision,recall,F1 score` are used

### Explainable AI Deployment

LIME and Grad-CAM techniques are used to provide interpretable visualizations of the model's predictions, enhancing the transparency and trustworthiness of the AI system

### ( I ) LIME for Explainable AI(XAI)
In order to enhance the transparency and interpretability of our CNN model, LIME `Local Interpretable Model-Agnostic Explanations` tool is a popular technique used in XAI to provide local explanations for the predictions made by complex machine learning models.

By applying LIME to the CNN model, it was able to generate human-understandable explanations for individual predictions to gain insights into how the model arrived at its decision, shedding light on the features and patterns that influenced the classification outcome.

### ( II ) Grad-CAM for Explainable AI(XAI)
In order to visualize the area on which the model focuses when making predictions in the form
of a heat map, intuitively showing the basis used by the model to make decisions.

The Grad-CAM `Gradient-Weighted Class Activation Mapping` method is a technique used
to visualize the regions of an input image that are important for predicting a particular class. It
highlights the regions of the image that contribute the most to the predictions made by the neural network.

### Website Development

A user-friendly web interface was developed using HTML-CSS to allow healthcare professionals to easily upload MRI images and receive diagnostic predictions along with interpretable visual explanations

![Home Page](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/home_page.png)

![Result Page](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/result_page.png)

## 04 Results and Discussion
The model was trained with a total of 5120 images for 10 epochs and tested with a test set of 1280 images. An accuracy of `98.60%` was achieved.

Results of LIME & Grad-CAM Deployment for Prediction

![Mild](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/mild.png)

![Moderate](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/moderate.png)

![Non-Demented](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/non.png)

![Very-Mild](https://github.com/JISHNU-2002/AL-ZHEIME-Main-Project/blob/main/images/verymild.png)

The results demonstrate the effectiveness of our approach in accurately diagnosing the stages of Alzheimer's disease. The deployment of explainable AI techniques provided valuable insights into the model's decision-making process.

## 05 Conclusion and Future Scope

This project successfully developed a deep learning model that enhances the early diagnosis of Alzheimer's disease. By integrating explainable AI, the system offers not only accurate predictions but also transparency in its operations, making it a valuable tool for healthcare professionals.

Future work could involve expanding the dataset, refining the model architecture, and integrating additional explainable AI techniques to further improve the system's accuracy and interpretability.

## 06 References
[1] Kokkula Lokesh, Nagendra Panini Challa, Abbaraju Sai Satwik, Jinka Chandra Kiran,Narendra Kumar Rao and Beebi Naseeba, (2023) Early Alzheimer’s Disease Detection Using Deep Learning, doi: 10.4108/eetpht.9.3966

[2] Sarasadat Foroughipoor,Kimia Moradi,Hamidreza Bolhasani,(2023) Alzheimer’s Disease Diagnosis by Deep Learning Using MRI-Based Approaches, https://doi.org/10.48550/arXiv.2310.17755

[3] Atefe Aghaei, Mohsen Ebrahimi Moghaddam, (2023) Smart ROI Detection for Alzheimer’s Disease prediction using explainable AI, https://doi.org/10.48550/arXiv.2303.10401
