import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class get_clothes_class():
    def __init__(self, model_path, encoded_word_dict_path, train_dataset_path):
        self.classes = ["outwear", "top", "trousers", "women dresses", "women skirts"]
        ### import train_dataset
        self.train_df = pd.read_csv(train_dataset_path)
        self.X_train = self.train_df.word_representation
        ### Import model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            f.close()
        ### Import encoding dictionary
        with open(encoded_word_dict_path, 'rb') as f:
            self.word_keys = pickle.load(f)
            f.close()
        ### Decoding dictionary
        self.raw_word_keys = {v:k for k,v in self.word_keys.items()}
    
    '''plaintext to encoded words format'''
    def _encode_input(self, raw_string):
        raw_string_list = raw_string.split()
        encoded_string_list = [self.word_keys[i.lower()] for i in raw_string_list if i.lower() in self.word_keys.keys()]
        return ' '.join(encoded_string_list)  

    '''encoded words to plaintext format'''
    def _decode_input(self, encoded_string):
        encoded_string_list = encoded_string.split()
        raw_string_list = [self.raw_word_keys[i.lower()] for i in encoded_string_list if i.lower() in self.raw_word_keys.keys()]
        return ' '.join(raw_string_list)  
    
    '''to process the encoded words for model prediction'''    
    def process_input(self, input_string):
        # Convert words into encoded input
        encoded_input_string = self._encode_input(input_string)
        print('encoded words: ', encoded_input_string)
        # # Tokenise and transform your encoded input
        processed_input=[encoded_input_string]
        # Pass the processed input into the prediction
        result = []
        for category in self.classes:
            print('... Processing {}'.format(category))
            # train the model using X_dtm & y
            self.model.fit(self.X_train, self.train_df[category])
            # compute the testing accuracy
            prediction = self.model.predict(processed_input)
            result.append(prediction)
        # #Get classes result
        detected_classes = []
        for idx in range(len(result)):
            for item in result[idx]:
                if item == 1:
                    detected_classes.append(self.classes[idx])

        output = list(set(detected_classes))
        

        # import pdb; pdb.set_trace()

        return output


## Example usage ###
# clothes=get_clothes_class('./finals_nlp_model.sav','./encoded_words.pkl', './TIL_NLP_train_dataset.csv')
# words = clothes._decode_input('w304300 w236555 w365489 w15393 w256905 w292000 w217871 w148650 w220790 w207614 w20894 w247655 w500010 w136109 w92187 w241910')
# print(words)
# result = clothes.process_input(words)
# print(result)

