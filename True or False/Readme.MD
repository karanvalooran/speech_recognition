# Speech Recognition - True or False Detection

This simple project aims at detecting the uttered word is "true" or "false".
The dataset is prepared for two classes: "true" and "false".
Each class has about 50 recordings with each one second long.

use "MFCC_feature_extraction.ipynb" to extract the MFCC features from the dataset.
The extracted features are stored in "datasetTorF.json".

use "True_or_False_speech_classification_cnn.ipynb" to train the features using CNN.
The trained model is stored "TrueorFalse_model.h5".

Testing accuracy of the model is 82.4%.
Training accuracy of the model is 89.3%.
Validation accuracy of the model is 84.0%.

use "speech_recognition_true_or_false.py" to test new sample.
 
### Speech Recognition Algorithm: CNN
### Audio Feature: MFCC (13 features)

## Python libraries used
1. librosa
2. keras
3. tensorflow
4. numpy
5. json
6. sklearn

