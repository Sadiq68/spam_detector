# spam_detector
The Files submited are spam_detector.py, model.txt, result.txt
The overall spam detection code is available in a single spam_detector python file.
The file consists of a class called NB_model and an object of the class stores the naive bias model generated out of train data.
The training data is picked up from the files available in train folder. So, if training data needs to be changed, the files in train folder should be modified.
The test data is picked up from the files provided in test folder. 
While training the model, a model.txt file is created by the NB_model object which stores all the words and their coresponding conditional probabilities
while testing the model, a result.txt file is generated which stores the output results for each test file
To run the code, once you have all the train and test data available in their respective folders, you just have to run the spam_detector python file (python spam_detector) which inturn generates the model.txt and result.txt in the same folder
