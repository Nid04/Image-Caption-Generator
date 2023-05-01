## CPS 595-P1 Software Engineering Project

University of Dayton

Department of Computer Science

CPS 595-P1 Software Engineering Project, Final Year

Instructor(s):
- Dr. Ahmed El Ouadrhiri
- Dr. Phu Phung 

## Software Engineering Project 

# Image Caption Generation using AI/ML with User Feedback

## Student Name: Nidhi Sinha (sinhan1@udayton.edu)

# Company Mentors

Arpan Bhattacharya; Rajanikanth Bhagavan Yadav; Anil Kumar

Synchrony Bank

# Project Management Information

Project Management board (Private access): https://trello.com/b/nzILGiVw/image-caption-generation-using-ai-ml/

Source code repository (Private access): https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/

# Table of Content #
1. [Overview] (#Overview)
2. [Background] (#Background)
3. [Project Context and Scope] (#Project Context & Scope)
4. [High-Level Requirements] (#High-level Requirements)
5. [Use case Diagram] (#Use Case Diagram)
6. [Sequence Diagram] (#Sequence Diagram)
7. [Implementation] (#Implementation)
8. [References] (#References)
10. [Appendix] (#Appendix)

## Overview ##
Image caption generation is a project in which a machine learning model is trained to generate natural language descriptions of an image. 
The goal is to create a model that can understand the content of an image and generate a caption that accurately describes it. 
There are several techniques that can be used for image caption generation, including neural networks and deep learning. 
The project involves utilizing a convolutional neural network (CNN) to extract features from an image and a recurrent neural network (RNN) to generate a caption. Upon completion of training, the model can then be used to generate captions for new images.

# Background #

Many articles, such as Image Captioning - A Deep Learning Approach [1], and Artificial Intelligence Based Image Caption Generation [2],  use the CNN-LSTM architecture for the Image Caption Generation model. In these works, the CNN model is used for the recognition and classification of an image whereas LSTM is capable of sequence prediction problem which identifiesthe next word.


**Basic of Image Captions**

![Figure 1: Basic of Image Captions](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Image%20files/Img1.png "Basic of Image Captions")
Figure 1: Basic of Image Captions

**Model Architecture Overview**

![Figure 2: Architecture of Captioning](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Image%20files/Img2.png "Architecture of Captioning")
Figure 2: Architecture of Captioning

The model consist of 3 different stages:
* Image Feature Extraction: In this stage the feature of an image is extracted by using VGG16 model. VGG16 is a working model for image detection. it consist of 16 layer which includes a pattern of 2 convolution layer and 1 dropout layer, it's fully connected at the end.

* Sequence Processor: This stage is used for manipulating text input that includes the word implanting layer. Later connected to the LSTM for the final stage of the image captioning.

* Decoder: The last stage combines the input which extracted fetures from an image and a sequence processing stage.

After successful generation of model will train the model on Flicker-8k dataset which have more that 8000 images and it's caption. Once the model is trained will test the model to generate the caption for new images and evaluate it with the actually caption for the image. In order to evaluate the machine translated text will use BLEU (Bilingual Evaluation). It lies between 0 to 1, where 0 means the generated caption doesn't overlap with the refernce text (low quality) and 1 means it perfectly overlap with the reference text (high quality). In the above article, the machine generated text was 60 to 70 percent accurate.


# Project Context & Scope #

The primary motivation for image caption generation is to enhance the accessibility and usability of visual content, especially for individuals with visual impairments. It can also be used to assist in content indexing and retrieval, as well as providing descriptive information for images in online publications, social media, and e-commerce websites.

The scope of image caption generation can range from generating basic captions that describe the content of an image to generating detailed and contextualized captions that convey emotions and sentiments. The applications can also vary, from educational and instructional content to entertainment and social media sharing. Overall, the image caption generation technology has the potential to improve the user experience and accessibility of visual content across various domains.

## High-level Requirements ##

Here are some high-level requirements for an Image Caption Generation project using AI/ML with user feedback:

* Datasets of images and their captions: It will be helpful to train and evaluate the model. The large dataset will help to evaluate the relationship between the image and it's caption and can generalize the images as well.
* Preprocessing: Preprocess the images to normalize their sizes and colors, and extract relevant features for input to the model.
* Image Caption Generation Model: Design and train a deep learning model, such as a convolutional neural network (CNN) and a recurrent neural network (RNN), to generate captions for the input images.
* User Feedback Component: Develop a user feedback component that allows users to provide feedback on the generated captions, which can be used to improve the accuracy of the model over time.
* User Interface: Design an intuitive and user-friendly interface that allows users to easily upload images, view generated captions, and provide feedback.
* Deployment: Deploy the image caption generation system on a reliable and scalable cloud platform, with the ability to handle large numbers of concurrent users and datasets.
* Performance: Optimize the model and system performance for fast and accurate image caption generation and feedback processing.


## Use Case Diagram ##

In the provided diagram, users have the option to submit an image and receive corresponding outcomes, as well as provide feedback on the results. By utilizing the "Upload Image" use case, users can upload their desired image, while the "Give Feedback" use case enables them to offer feedback on the generated tags. The "Display Output" use case displays the generated tags and corresponding confidence scores. The algorithm utilizes the image to extract relevant features and text, and can learn from user comments to improve its performance.

![Figure 3: Use Case Diagram](https://user-images.githubusercontent.com/90881345/235277614-afd1f11f-f3ca-4eee-b46d-98bc98d9e1f7.png "Use Case Diagram")
Figure 3: Use Case Diagram

## Sequence Diagram ##

The given image depicts a scenario where a user interacts with a web application to upload an image and provide feedback. The Image Captioning Model then processes the uploaded image to extract relevant features and text, generate corresponding tags, and calculate confidence scores. Upon viewing the results on the web application, the user can offer feedback, which in turn helps to enhance the accuracy of the Image Captioning Model.

![Figure 4: Sequence Diagram](https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY/blob/main/Sequence_Diagram.png "Sequence Diagram")

Figure 4: Sequence Diagram

# Implementation #

The project comprises of two primary components: one focuses on developing and training the image caption generation model, while the other concentrates on designing the user feedback component.

## Caption Generation Model ##

The model is trained on a dataset of images and corresponding captions and can generate captions for new images that it has not seen before.

**Dataset**
 
The dataset used for this project is the Flickr8k [3] & Flickr30k [4] dataset, which contains 8,000 images and five captions per image.
The dataset is preprocessed by extracting the features from the images using a pre-trained VGG16 model and cleaning and tokenizing the captions.

**Model**

The model used for this project is a deep neural network consisting of an image encoder and a caption decoder. The image encoder uses a pre-trained VGG16 model to extract features from the input image. The caption decoder consists of an embedding layer, an LSTM layer, and two dense layers.

**Prerequisites**

* Python 3.9.6
* Tensorflow 2.11.0
* Pickle
* tqdm
* NumPy 1.21.5

## User Feedback Feature ##

User Feedback feature will help to get the feedback from user end for fine tuning our image captioning model. It collects and store the user given captions and store it into a dataset. Then, preodically need to retrain the existing model for enhanced prediction. To implement this feature we use Human-in-the-loop learning algorithm. This Machine Learning approach involving human expertise for adaptable, ethical decision-making and continuous model improvement. The user inputs helps our image captioning model for continuous learning and optimization.

**User Feedback Architecture**

The below image will show the implementation of the user feedback concept in our image captioning application.

![Figure 5: User Feedback Architecture](https://user-images.githubusercontent.com/90881345/235231521-5ca1be9c-78fb-44ac-ba49-dfa0f1e85ba2.png "User Feedback Architecture")
Figure 5: User Feedback Architecture

## User Interface Installation and Setup ##

The web application built using the Django web framework for uploading an image & to provide feedback caption on image.

**Prerequisites**
* Python 3.9.6
* Django 4.1.6
* pip (Python Package Installer)

**Step 1: Clone the repository**
```
git clone https://github.com/Nid04/Image-Caption-Generation---SYNCHRONY
```

Navigate to the project directory:
```
cd Image-Caption-Generation---SYNCHRONY
```

**Step 2: Install Required Dependencies**

Install the required dependencies using pip:
```
pip install -r requirements.txt
```

**Step 3: Run the Application**

Navigate to mysite directory:
```
cd mysite
```

Run the command:
```
python manage.py migrate
```
```
python manage.py runserver
```

Open the highlighted URL in your browser
![image](https://user-images.githubusercontent.com/90881345/235221052-8cda8d93-2fd3-470c-b330-56bb4a8330ad.png)
Figure 6: Console


## Demo ##

* Screenshot 1: A user interface for uploading an image and get caption

<img width="843" alt="image" src="https://user-images.githubusercontent.com/90881345/232843583-58b35760-5976-47f2-bd01-4a315f43054a.png">

* Screenshot 2: After the user has chosen the image for which they would like to generate a caption and click on upload button.

<img width="843" alt="image" src="https://user-images.githubusercontent.com/90881345/235280107-2e4b4862-c719-4794-a854-5cd69ca9d611.png">

* Screenshot 3: Upon clicking the upload button, the UI loads the image and generates the corresponding caption at the bottom of the image.

<img width="843" alt="image" src="https://user-images.githubusercontent.com/90881345/232844510-fe2136e6-6c30-4bf6-a5b7-53412731972b.png">

* Screenshot 4: The user has the option to offer feedback for the uploaded image by clicking on the submit button.

<img width="843" alt="image" src="https://user-images.githubusercontent.com/90881345/232853507-e99596f9-1c67-49d1-ae06-18e355053cb5.png">

* Screenshot 5: After the user clicks the submit button, the entered data is saved to the dataset. Then, the user will receive a success or failure message indicating whether the submission was completed successfully or not.

<img width="843" alt="image" src="https://user-images.githubusercontent.com/90881345/232853612-625352c4-7818-4d1f-a9d1-075b5281739e.png">

# References #

1. Image Captioning - A Deep Learning Approach: http://www.ripublication.com/ijaer18/ijaerv13n9_102.pdf
2. Artificial Intelligence Based Image Caption Generation: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3648847
3. https://www.kaggle.com/datasets/adityajn105/flickr8k
4. https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
5. https://www.deeplearning.ai/courses/
6. https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

# Appendix #

<pre>
mysite : This is the main project directory
   |-- .DS_Store
   |-- captionapp 
   |   |-- .DS_Store
   |   |-- __init__.py
   |   |-- admin.py : This is where you register models to include them in the Django administration site—using this site is optional.
   |   |-- apps.py : This includes the main configuration of the captionapp application.
   |   |-- forms.py : It is where the django documentation recommends you place all your forms code; to keep your code easily maintainable.
   |   |-- migrations : This directory will contain database migrations of your application. Migrations allow Django to track your model changes and synchronize the database accordingly.
   |   |-- models.py : This includes the data models of the application; all Django applications need to have a models.py file, but this file can be left empty.
   |   |-- static : contain style sheet
   |   |   |-- .DS_Store
   |   |   |-- css  
   |   |   |   |-- main.css
   |   |-- templates : contain html file
   |   |   |-- base.html : main template for all the pages
   |   |   |-- upload
   |   |   |   |-- upload.html : upload page template
   |   |-- tests.py : This is where we can add tests for the application.
   |   |-- urls.py : A request in Django first comes to urls.py and then goes to the matching function in views.py
   |   |-- views.py : The logic of the application goes here; each view receives an HTTP request, processes it, and returns a response.
   |-- db.sqlite3
   |-- manage.py : This is a command-line utility used to interact with the project.
   |-- media : All the uploaded images saved inside images folder
   |   |-- images
   |-- mysite
   |   |-- __init__.py : An empty file that tells Python to treat the mysite directory as a Python module.
   |   |-- asgi.py : This is the configuration to run the project as ASGI, the emerging Python standard for asynchronous web servers and applications.
   |   |-- settings.py : This indicates settings and configuration for the project and contains initial default settings.
   |   |-- urls.py : This is the place where the URL patterns live. Each URL defined here is mapped to a view.
   |   |-- wsgi.py : This is the configuration to run the project as a Web Server Gateway Interface (WSGI) application.
   |-- user_dataset.csv : It contains name of the image and the feedback given by the user
requirements.txt : contains all the required packages
retrain_model.py : It contains code to retrain the model based on user feedback
schedule_retrain.py : It contains code to run the retrain_model.py code preodically
</pre>
