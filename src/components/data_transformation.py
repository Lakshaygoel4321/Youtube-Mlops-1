import sys
import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from src.logger import logging
from src.exception import USvisaException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact

nltk.download('stopwords')

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def clean_text(text):
        try:
            if isinstance(text, str):
                text = text.lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('[^a-zA-Z]', ' ', text)
                text = re.sub('\s+[^a-zA-Z]\s+', '', text)
                text = re.sub('\w*\d\w*', '', text)
                stopword = set(stopwords.words("english")) - {"not","no","however","but"}
                text = ' '.join([word for word in text.split() if word not in stopword])
                # pattern = re.compile(r'\b(' + '|'.join(stopwords.words('english')) + r')\b\s*')
                # text = re.sub(pattern, '', text)
            return text
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def use_porter(text):
        try:
            ps = PorterStemmer()
            text = str(text)  # Convert to string first
            return ' '.join([ps.stem(word) for word in text.split()])
        except Exception as e:
            raise USvisaException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            

            logging.info("Reading train and test files")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            for df in [train_df, test_df]:
                df["clean_comment"] = df["clean_comment"].apply(self.clean_text)
                df["clean_comment"] = df["clean_comment"].apply(self.use_porter)

            logging.info("Text cleaning and stemming completed")

            # Tokenizer & Padding
            # tokenizer = Tokenizer(num_words=15000)
            # tokenizer.fit_on_texts(train_df["clean_comment"])

            # x_train_seq = tokenizer.texts_to_sequences(train_df["clean_comment"])
            # x_test_seq = tokenizer.texts_to_sequences(test_df["clean_comment"])

            # x_train = pad_sequences(x_train_seq, maxlen=150)
            # x_test = pad_sequences(x_test_seq, maxlen=150)

            
            vectore = CountVectorizer(max_features=1000)
            x_train_vectore = vectore.fit_transform(train_df['clean_comment']).toarray()
            x_test_vectore = vectore.transform(test_df['clean_comment']).toarray()

            y_train = train_df[TARGET_COLUMN].values
            y_test = test_df[TARGET_COLUMN].values

            random = RandomOverSampler()
            # for train
            x_train_res,y_train_res = random.fit_resample(x_train_vectore,y_train)
            # for test
            x_test_res,y_test_res = random.fit_resample(x_test_vectore,y_test)

            # Save arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, np.c_[x_train_res, y_train_res])
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, np.c_[x_test_res, y_test_res])

            # Save tokenizer
            save_object(self.data_transformation_config.transformed_object_file_path, vectore)

            logging.info("Data transformation completed and objects saved")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise USvisaException(e, sys)