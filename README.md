# SmartClass-A.I.ssistant  



# Group name:  TBD


Members:  
- Matthew Flaherty 40228462
- Lauren Rigante 40188593
- Justin Cheng 40210279


# Files:  
Part 1:
- remove_duplicates.py: This file removes all images with duplicate content. It computes the MD5 hash of each image and removes the most similar images
- check_and_remove_corrupted_images.py: This file checks if it can open all images in a directory. If an image cannot be opened, it is considered corrupted and deleted from its directory.
- resize_images.py: This file changes all images to a standard 224x224 images size. The method ensures that the face in each image is centered by detecting faces using a Haar cascade classifier, calculating the center of the detected face, and then cropping and resizing the image around that center point.

Part 2:  

Part 3:  



# Steps to execute code:  
Data Cleaning: 
- First, we downloaded images from the 2 sources and combined them into 4 folder, 1 for each class (Happy, Angry, Neutral and Focused).
- Then, we ran the remove_duplicates.py to remove all duplicate images.
- Next, the check_and_remove_corrupted_images.py was used to remove corrupted images.
- After that we ran the resize_images.py to make all images a 224x224 standard size, and use a Haar cascade classifier to center the images around the first face detected.
- We then manually removed some images, for example images that contained:
  - Water marks
  - No face
  - Face covered by an object
- Then, we added our own faces to each class
-  DO WE TALK ABOUT REMOVING IMAGES WHERE A PERSON RE APPEARS??
-  Finally the images were cleaned and ready to move on to the next step.

Data Visualization: 





# DATA SOURCES:
- Natural Human Face Images for Emotion Recognition - https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition/data?select=happiness
- 6 Human Emotions for image classification - https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier
