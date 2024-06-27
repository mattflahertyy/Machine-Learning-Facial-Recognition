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
- change_index.py: This changes the index values in each json from from 1-4 based to 0-3 based.
- evaluate_single_image.py: Does the same as above but only evaluates one image.
- change_brightness.py: Checks if an image isn't brught enough, if so then it enhances the brightness.

Part 3:
- detect_bias_gender.py: This file takes the existing CNN architecture and verifies the bias with the metric of gender based on the labels in the csv file
- detec_bias_race.py: This file takes the existing CNN architecture and verifies the bias with the metric of race based on the labels in the csv file
- cnn_model_evaluation.py: Same as the previous model evaluation from part 2, except it only evaluates the main model
- cnn.py: Same as the previous cnn.py from part 2
- split_csv.py: Same as previous code that splits csv file into train, test, and validation data
- print_csv_stats.py: Same as previous code taht prints the stats of the csv files containing the data
- k_fold.py: This file runs a 10 fold cross validation on the main model on the same seed while printing the stats such as accuracy, precision, recall, and f1 score for each of the folds and then the average.
- convert_random_grayscale.py: This file takes a random image and converts it to grayscale
- decrease_brightness.py: This file goes through the images and if it passes a certain threshold, it will decrease the brightness
- find_brightness.py: This file will calculate the brightness of all images and list their values in descending order
- increase_brightness.py: This file goes through the images and if it passes a certain threshold, it will increase the brightness



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
- First we renamed all images just for our own personal understanding when dealing with the images.  
- Then we ran the splitting_data.py file which created a 3 JSON files for train, validation and test and it stored the image names along with their label number from 1-4.  
- Next we ran json_to_csv.py which converted the 3 json files to csv, and stored the image name, image path, label name and label number.
- After realizing we should have our 1-4 based indexes 0-3 based, we ran change_index.py.
- Next it was time to run the cnn.py file, this trained our regular model and stored the best model for the epoch with the lowest validation loss. This regular model was 4 convolutional layers with 3x3 kernel size.
- After this it was time for variant 1 and 2 (cnn_variant1.py and cnn_variant2.py), both of these were similar to the cnn.py except variant 1 had 5 layers with 5x5 kernel size and variant 2 had 3 layers with 2x2 kernel size.
- Finally we ran the cnn_model_evaluation.py file which took our best 3 models and used the test set to measure the performance, calculating the accuracy, precision, recall, and f1 measure for both macro and micro.
- After this we ran the evaluate_single_image.py which does the same as above but analyzes only one image.
- At the end we decided to add the change_brightness.py which modified the brightness of images if they were too low, resulting in about 10% increase in accuracy.

Data Evaluation:
- First we created code to perform 10 fold cross valuation, and all the separated images were copied into a single csv file
- Then we ran the k_fold.csv which ran the 10 fold cross valuation. It split the code into 10 parts and evaluated each as the test set and splits the train dataset into train and validation subsets.
- Next the code evaluated the metrics for micro and macro accuracy, precision, recall, f1 score and the average of the 10 folds of each metric.

Data Augmentation:
- First we manually added new tags to all images with the 2 chosen additional biases
- Then we ran the detect_bias_gender.py and detect_bias_race.py to determine the biases that existed within the data
- The data was further augmented by adding more data to the asian, black and hispanic categories.
- The new images were put through the convert_random_grayscale.py, decrease_brightness.py, find_brightness.py and increase_brightness.py to ensure consistency with the prior images.
- The data was evaluated again. 

# DATA SOURCES:
- Natural Human Face Images for Emotion Recognition:
  - https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition/data?select=happiness
- 6 Human Emotions for image classification:
  - https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier
- Facial Emotion Recognition:
  - https://www.kaggle.com/datasets/chiragsoni/ferdata/data
