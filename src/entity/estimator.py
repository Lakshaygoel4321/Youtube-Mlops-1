import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import USvisaException
from src.logger import logging
import pickle


# class TargetValueMapping:
#     def __init__(self):
#         self.Certified:int = 0
#         self.Denied:int = 1
#     def _asdict(self):
#         return self.__dict__
#     def reverse_mapping(self):
#         mapping_response = self._asdict()
#         return dict(zip(mapping_response.values(),mapping_response.keys()))
    
# from tensorflow.keras.preprocessing.sequence import pad_sequences

class USvisaModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object  # This is a Keras Tokenizer
        self.trained_model_object = trained_model_object  # This is your LSTM model

    def predict(self, dataframe: DataFrame):
        logging.info("Entered predict method of USvisaModel class")

        try:
            logging.info("Using tokenizer to convert text to sequences")
            
            texts = dataframe['tweet'].tolist()  # column name should match your input
            
            with open("vectore.pkl",'rb') as file:
                vectore = pickle.load(file)
                
            #sequences = self.preprocessing_object.texts_to_sequences(texts)
            transformed_feature = vectore.transform(texts)
            #transformed_feature = pad_sequences(sequences, maxlen=150)  # same maxlen used during training

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise USvisaException(e, sys) from e