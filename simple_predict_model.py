import json
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch import BoolTensor
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from scipy.special import softmax

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

    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output_bert = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_bert[0]
        pooler = hidden_state[:, 0, :]  # only take the CLS one
        output = self.classifier(pooler)

        if labels is None:
            return output
        # end

        loss = self.func_loss(output.view(-1, self.num_labels), labels.view(-1))
        return (loss, output)
    # end

    def predicts(self, samples_input, need_raw=True):

        if type(samples_input) is str:
            samples_input = [samples_input]
        # end

        outputs = []
        for sample_input in samples_input:
            sentence = ' '.join(sample_input.split())
            inputs = self.tokenizer.encode_plus(
                sentence, None,
                add_special_tokens=True,
                max_length=self.config_classifier.get('bert').get('max_length'),
                padding='max_length',
                return_token_type_ids=True,
                truncation=True,
                return_tensors='pt'
            )

            for key in inputs:
                inputs[key].to(self.device)
            # end

            with torch.no_grad():
                output = self(**inputs).cpu().numpy().flatten()
                outputs.append(output)
            # end
        # end

        labels = self.labels_output_classifier

        if need_raw:
            info_result = {
                'outputs': [output.tolist() for output in outputs],
                'labels': labels
            }
        else:
            result_softmax = softmax(np.array(outputs), axis=1) # transform to confidence

            # pick the highest confidence item and corresponding label
            list_conf = np.amax(result_softmax, axis=1, keepdims=True).reshape(-1).tolist()
            list_index_label = np.argmax(result_softmax, axis=1).tolist()
            labels_result = [labels[index_label] for index_label in list_index_label]
            info_result = {
                'outputs': [[conf, label] for conf, label in zip(list_conf, labels_result)],
                'labels': None
            }
        # end
        
        return info_result
    # end

# end

path_model_1 = os.path.join('models', 'bert', 'target_v1')
model_1 = SimpleBertClassifier(path_model_1)
model_1.load(is_eval=False)
print('model 1 loaded')

# prediction
print(model_1.predicts('hello world', need_raw=False)) # -> copied from gosv-ai-pipeline/pkg/logics.py#_label_results_and_confidences
print(model_1.predicts('hello world', need_raw=True)) # -> same in modeldock