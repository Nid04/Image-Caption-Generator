# CPS 595-P1 Software Engineering Project #

University of Dayton
Department of Computer Science

Instructor(s):
* Dr. Phu Phung 
* Dr. Ahmed El Ouadrhiri

This is the repository for the CPS 595-P1 Software Engineering Project.

Name: Nidhi Sinha

Email: sinhan1@udayton.edu

# Image Caption Generation using AI/ML #

# Company #
* Name: Synchrony

# Project Management Information #
Management board (Private access): https://trello.com/b/nzILGiVw/image-caption-generation-using-ai-ml/

Source code repository (private access): https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/

# Overview #
Image caption generation is a task in which a machine learning model is trained to generate natural language descriptions of an image. 
The goal is to create a model that can understand the content of an image and generate a caption that accurately describes it. 
There are several techniques that can be used for image caption generation, including neural networks and deep learning. 
One popular approach is to use a convolutional neural network (CNN) to extract features from the image, and a recurrent neural network (RNN) to generate the caption. 
Once the model is trained, it can be used to generate captions for new images.

# Background #

In many articles such as Image Captioning - A Deep Learning Approach, Artificial Intelligence Based Image Caption Generation, etc used the CNN-LSTM architecture for Image Caption Generation model. In which the CNN is used for recognition and classification of an image where as LSTM is capable of sequence prediction problem which identifies the next word.

**Basic of Image Captions**

![image](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Image%20files/Img1.png)

**Model Architecture Overview**

![image](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Image%20files/Img2.png)

The model consist of 3 different stages:
* Image Feature Extraction: In this stage the feature of an image is extracted by using VGG16 model. VGG16 is a working model for image detection. it consist of 16 layer which includes a pattern of 2 convolution layer and 1 dropout layer, it's fully connected at the end.

* Sequence Processor: This stage is used for manipulating text input that includes the word implanting layer. Later connected to the LSTM for the final stage of the image captioning.

* Decoder: The last stage combines the input which extracted fetures from an image and a sequence processing stage.

After successful generation of model will train the model on Flicker-8k dataset which have more that 8000 images and it's caption. Once the model is trained will test the model to generate the caption for new images and evaluate it with the actually caption for the image. In order to evaluate the machine translated text will use BLEU (Bilingual Evaluation). It lies between 0 to 1, where 0 means the generated caption doesn't overlap with the refernce text (low quality) and 1 means it perfectly overlap with the reference text (high quality). In the above article, the machine generated text was 60 to 70 percent accurate.


# Project Context & Scope #
The context for an Image Caption generation using AI/ML project is to automatically generate natural language descriptions for images.

The scope of this project will include the following steps:

* Collecting, cleaning and preprocessing a dataset of images and their corresponding captions.
* Design and implement a model for image caption generation.
* Train the model on the dataset.
* Evaluation of the model on the dateset.
* Optimizing the model.
* Integartion of the model with the main application.

The main objective of our project is to generate captions that accurately describe an image.

## High-level Requirements ##

The high level requirements for an Image Caption generation using AI/ML project are:

* Dataset of an image and it's caption: It will be helpful to train and evaluate the model. The large dataset will help to evaluate the relationship between the image and it's caption and can generalize the images as well.
* ML model: The model will extract feature and generate caption based on the feature.
* Development environment: It includes the hardware & software which will help to train, evaluate & optimize our model
* Evaluation & tuning of model
* Integration of model with large system or an application

## Use Case Diagram ##
![ImageCaptionGeneration Usecase Diagram](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Usecase_diagram.png)

As shown in the diagram, the user can either submit an image and check the outcomes as well as can give feedback. The user can upload an image using the "Upload Image" use case, and they can input a caption using the "Enter Caption" use case which allow user to offer feedback on the outcomes, the use case "View Results" shows the generated tags and confidence scores. The algorithm derives tags and confidence scores, pulls features and text from the image, and may learn from user comments.

## Sequence Diagram ##
![Image Caption Generation - Sequence Diagram](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Sequence_Diagram.png)

In this illustration, the user uses a web application to upload an image and provide a caption. The Image Captioning Model receives the image and caption from the web application and uses them to extract text and features, create tags, and calculate confidence scores. The user can give input after viewing the findings on the online application. The user feedback will help the image captioning model become more accurate.

# References #
* Image & Caption dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
* https://www.deeplearning.ai/courses/
* https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
* Image Captioning - A Deep Learning Approach: http://www.ripublication.com/ijaer18/ijaerv13n9_102.pdf
* Artificial Intelligence Based Image Caption Generation: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3648847
