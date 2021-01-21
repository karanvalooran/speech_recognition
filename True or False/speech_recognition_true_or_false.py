#importing the libraries
import tensorflow.keras as keras
import numpy as np
import librosa
import math

#Specifying the model path
MODEL_PATH = "/content/TrueorFalse_model.h5"

#audio parameters
NUM_SAMPLES_TO_CONSIDER = 22050 #1 second audio
DURATION = 1
SAMPLE_RATE= 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


class _trueorfalse_detection:
  model =None
  _mappings = [
        "true",
        "false"
    ]
  _instance  = None

  def predicter(self, file_path):

    #extract mfcc's
    MFCCs = self.preprocess(file_path)
    MFCCs = MFCCs[np.newaxis,..., np.newaxis]

    #make the predictions
    predictions = self.model.predict(MFCCs)
    predicted_index = np.argmax(predictions)
    predicted_keyword = self._mappings[predicted_index]

    return predicted_keyword

  def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length = 512, num_segments = 5):
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK /num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)
    
    #load audio file
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    for s in range(num_segments):
        start_sample = num_samples_per_segment *s
        finish_sample = start_sample + num_samples_per_segment
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
          signal = signal[:NUM_SAMPLES_TO_CONSIDER]

    #extract MFCCs
        MFCCs = librosa.feature.mfcc(signal[start_sample:finish_sample], n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return MFCCs.T


def Speech_detection():

  if _trueorfalse_detection._instance is None:
    _trueorfalse_detection._instance = _trueorfalse_detection()
    _trueorfalse_detection.model = keras.models.load_model(MODEL_PATH)
  
  return _trueorfalse_detection._instance


if __name__ =="__main__":
  sp = Speech_detection()
  solution = sp.predicter("/content/test1.wav")
  print(f'Predicted output: {solution}')

