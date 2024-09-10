# LBPH Face Recognition using OpenCV
## Modules and libraries required :
1. OpenCV : in terminal;
   - > pip install opencv-contrib python
2. Numpy : in terminal;
   - > pip install numpy
3. Pillow : in terminal;
   - > pip install pillow
***NOTE :*** in-built os module in Python
## Additional installations :
1. A parent repository - FaceDetection - with four child folders for dataset collection, traineing the cascade, a trained .yaml extension, and for recog.
2. The xml-code for implementing the desired cascade from OpenCV's cascades_github page (can implement in VS's .xml extenison)

 ## Collection.py
 - File to create a dataset of greyscale images for the haar cascade to train on
Capture successive greyscale images from the cv.VideoCapture(0) function into image variables, convert to greyscale using the cv.cvtColor('img_var',
COLOR_RGB2GRAY)
Write the image paths into the collection dataset repo.
***NOTE:*** resize all images to the same dimensions

## Training.py
- Initialise the LPBH cascade into a variable via the function cv.CascadeClassifier('absolute path to .xml file goes here')
 Use the Image module from Pillow to work directly with folder-file paths, and access images directly from the collections' folder into a for loop via the Image.oepn() function
With concurrent ID and counter arrays, pass every sample of each image into one loop iteration for training, via the cv.train('img array', 'id array') function
Save the trained LPBH into another cascade variable via *cascadevar*.write() function

## Trained.yaml
- A tangible output of the trained cascade to check for potential (though unlikely) errors, in matrix-form

## Recognize.py
- Copy the trained cascade path into a variable via the .copy() method, and initialise another cascade variable as described in -> Training.py
Using a simple counter, feed the image and IDs arrays into the reognizer-variable (the same which was previously trained);
***NOTE:*** must convert IDs list into a proper array using the numpy.array() method
Use the.predict() method to finally get results; tangible output on the display must make use of the cv.rectangle() and cv.putText() functions
