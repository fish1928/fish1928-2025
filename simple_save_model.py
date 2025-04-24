import json
import os
import sys
from datetime import datetime
import pandas as pd
import torch
from torch import BoolTensor
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SimpleBertClassifier(torch.nn.Module):

    DEFAULT_FILENAME_CLASSIFIER = '.model.json'
    DEFAULT_FILENAME_BERT = 'bert_config.json'
    DEFAULT_FILENAME_MODEL = 'model.pt'
    DEFAULT_KEYS_IGNORED_CLASSIFIER = ['metrics', 'allmetrics']

    def __init__(self, path_folder_model=None):
        super(SimpleBertClassifier, self).__init__()

        filename_config_classifier = self.__class__.DEFAULT_FILENAME_CLASSIFIER
        filename_config_bert = self.__class__.DEFAULT_FILENAME_BERT
        filename_model = self.__class__.DEFAULT_FILENAME_MODEL
        keys_ignored_classifier = self .__class__.DEFAULT_KEYS_IGNORED_CLASSIFIER


        self.path_folder_model = path_folder_model
        self.path_config_bert = os.path.join(path_folder_model, filename_config_bert)
        self.path_config_classifier = os.path.join(path_folder_model, filename_config_classifier)
        self.path_file_model = os.path.join(path_folder_model, filename_model)

        with open(self.path_config_classifier, 'r') as file:
            self.config_classifier = json.load(file)
        # end

        for key in keys_ignored_classifier:
            if key in self.config_classifier:
                del(self.config_classifier[key])
            # end
        # end

        # classfier parameters
        self.classifier_input_size = self.config_classifier.get('bert').get('input_size')
        self.classifier_max_length = self.config_classifier.get('bert').get('max_length')
        self.classifier_output_size = self.config_classifier.get('bert').get('output_size')

        self.labels_output_classifier = self.config_classifier.get('classes')
        self.dict_label_index = {label: index for index, label in enumerate(self.labels_output_classifier)}
        self.num_labels = len(self.dict_label_index)
        # classifier parameters done

        self.config_l1 = None
        self.l1 = None
        self.linear = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loaded = False

        self.func_loss = None
    # end


    def load(self, is_eval=True):
        if not self.loaded:
            self.config_l1 = BertConfig.from_pretrained(self.path_config_bert)
            self.l1 = BertModel(self.config_l1)
            self.classifier = torch.nn.Linear(self.classifier_input_size, self.classifier_output_size)
            
            self.load_state_dict(torch.load(self.path_file_model, map_location=torch.device(self.device)))

            print('Please Ignore warning message sent by BertTokenizer below')
            self.tokenizer = BertTokenizer.from_pretrained(self.path_folder_model)
             
            if is_eval:
                self.eval()
            else:
                self.func_loss = torch.nn.CrossEntropyLoss()
                self.train()
            # end
            self.loaded = True
        return self
    # end

    def save(self, path_to_save):
        self.tokenizer.save_pretrained(path_to_save)
        print('[SUCCESS] tokenizer saved to {}.'.format(path_to_save))

        path_to_save_config = os.path.join(path_to_save, self.__class__.DEFAULT_FILENAME_BERT)
        with open(path_to_save_config, 'w+') as file:
            file.write(self.config_l1.to_json_string())
        # end
        print('[SUCCESS] l1 config saved to {}.'.format(path_to_save_config))

        path_to_save_model = os.path.join(path_to_save, self.__class__.DEFAULT_FILENAME_MODEL)
        torch.save(self.state_dict(), path_to_save_model)
        print('[SUCCESS] l1 model saved to {}.'.format(path_to_save_model))

        path_to_save_classifier = os.path.join(path_to_save, self.__class__.DEFAULT_FILENAME_CLASSIFIER)
        with open(path_to_save_classifier, 'w+') as file:
            file.write(json.dumps(self.config_classifier))
        # end
        print('[SUCCESS] classifier config saved to {}.'.format(path_to_save_classifier))
    # end
# end



# sample of loading model and saving model
version_model = 'target_v3'

path_model = os.path.join('models','bert',version_model)
classifier = SimpleBertClassifier(path_model)
classifier.load(is_eval=False)
print('loaded')

version_model_target = version_model[:-1] + str(int(version_model[-1]) + 1)
path_model_save = os.path.join('models','bert',version_model_target)
os.makedirs(path_model_save, exist_ok=True)

classifier.save(path_model_save)