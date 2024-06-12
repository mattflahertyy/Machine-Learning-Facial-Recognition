# SmartClass-A.I.ssistant  



# Group name:  AI-cademic Weapons


Members:  
- Matthew Flaherty 40228462
- Lauren Rigante 40188593
- Justin Cheng 40210279


# Files:  
Part 1:
- remove_duplicates.py: This file removes all images with duplicate content. It computes the MD5 hash of each image and removes the most similar images
- check_and_remove_corrupted_images.py: This file checks if it can open all images in a directory. If an image cannot be opened, it is considered corrupted and deleted from its directory.
- resize_images.py: This file changes all images to a standard 224x224 images size. The method ensures that the face in each image is centered by detecting faces using a Haar cascade classifier, calculating the center of the detected face, and then cropping and resizing the image around that center point.
- split_data.py: Once we had at least 500 images in each of the 4 class folders, this file split the images into test and train folders (80% for train and 20% for test). Like angry_train, angry_test, happy_train, etc.
- rename_train_images.py: LABELLING - This file changes the names of all images in the train folders to angry_1.png, angry_2.png, etc.
- class_distribution.py: Create a bar graph showing the number of images in each class with each graph labeled correctly. It uses the imports os and matplotlib to access the files and then plot them in a bar graph. 
- pixel_intensity_distribution.py: This file plots the aggregated pixel intensity distribution for a specified class. The user may choose the specific class in the code. The function reads the images in the class, aggregates the pixel intensities, and then plots them onto the table using matplotlib.
- sample_images.py: This code loads 15 random images and aggregates the pixel intensity distribution for the selected images. Then it displays the sample images with the pixel intensity graphs next to each other.

Part 2:  
- label_images.py: This file renamed the images based on their classname depending on the directory they are stored in. It doesn't actually do labelling, just renaming.
- splitting_data.py: This file split the images into a train, validation and test set (70%, 15%, 15%) and store it into separate json files along with their label from 1-4 based on classname.
- json_to_csv.py: This file converts the json to csv which contains the image name, image path, label name and label number.
- cnn.py: This file is for our regular model. First we loaded the training and validation sets using DataLoader, then created 4 layers with max pooling after each layer, along with 3x3 kernel sizes. This file trains the model, and for each epoch it calculates the training and validation loss. If the validation loss reaches a peak, the program has a patience of 3 so it will always keep the best scoring model out of max 15 epochs. There is early stoppage, so if the validation loss does best its lowest score after 3 epochs it will stop the program.
- cnn_variant1.py: This was the same as the regular CNN model, but it is 5 layers with 5x5 kernel.
- cnn_variant2.py: This was the same as the regular CNN model, but it is 3 layers with 2x2 kernel.
- cnn_model_evaluation.py: This file loads the 3 models (regular, variant 1 and variant 2), and evaluates them using the test set. It calculates the accuracy, precision, recall, f1 measure and the macro and micro precision. It also plots a confusion matrix for each of the 3 models.

Part 3:  TBD



# Steps to execute code:  
Data Cleaning: 
- First, we downloaded images from the 2 sources and combined them into 4 folder, 1 for each class (Happy, Angry, Neutral and Focused).
- Then, we ran the remove_duplicates.py to remove all duplicate images.
  - This file uses MD5 hashing to remove images that look similar to ensure no person appears twice.
- Next, the check_and_remove_corrupted_images.py was used to remove corrupted images.
- After that we ran the resize_images.py to make all images a 224x224 standard size, and use a Haar cascade classifier to center the images around the first face detected.
- We then manually removed some images, for example images that contained:
  - Water marks
  - No face
  - Face covered by an object
- Then, we added our own faces to each class
-  Finally the images were cleaned and ready to move on to the next step.
-  Once we had our 4 folders for each class - happy, angry, neutral and focused - we ran the split_data.py to split into 8 new folders (happy_train, happy_test, angry_train, etc).
-  Then we ran the rename_train_images.py to label all train images to angry_1.png, angry_2.png, etc
-  Once we were done with this, we create 2 folders called final_train and final_test, and moved each class of images into their respective folders

Data Visualization: 

- Created a bar graph showing the number of images in each class with each graph labeled correctly.
  - Used the imports os and matplotlib to access the files and then plot them in a bar graph. 
- Plotted the aggregated pixel intensity distribution for a specified class.
  - Read the images in the class
  - Aggregated the pixel intensities
  - Plotted them onto the table using matplotlib
- Loaded 15 random images and aggregated the pixel intensity distribution for the selected images.


Training the model:



# DATA SOURCES:
- Natural Human Face Images for Emotion Recognition - https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition/data?select=happiness
- 6 Human Emotions for image classification - https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier
- Facial Emotion Recognition - https://www.kaggle.com/datasets/chiragsoni/ferdata/data
