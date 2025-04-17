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


MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 512

FILENAME_TEST = 'test.csv'
DIR_OUTPUT = 'results'

DEVICE_DEFAULT = 'cuda'

def get_ts():
    return datetime.utcnow().replace(microsecond=0).isoformat()
# end


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    # end
# end

def read_passages(path_data, path_label, test_size=0):
    df = pd.read_csv(path_data)

    documents = df['processed'].to_list()
    labels_str = df['target'].to_list()

    samples = documents

    with open(path_label, 'r') as file:
        labels_list = sorted(json.load(file))
    # end

    labels_all = {l: idx for idx, l in enumerate(labels_list)}

    labels = [labels_all[label_str] for label_str in labels_str]

    if test_size > 0:
        return train_test_split(samples, labels, test_size=test_size, stratify=labels, random_state=234), labels_list
    else:
        return (samples, samples, labels, labels), labels_list
    # end
# end

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
            config_classifier = json.load(file)
        # end

        for key in keys_ignored_classifier:
            del(config_classifier[key])
        # end

        # classfier parameters
        self.classifier_input_size = config_classifier.get('bert').get('input_size')
        self.classifier_max_length = config_classifier.get('bert').get('max_length')
        self.classifier_output_size = config_classifier.get('bert').get('output_size')
        #

        self.labels_output_classifier = config_classifier.get('classes')
        self.dict_label_index = {label: index for index, label in enumerate(self.labels_output_classifier)}
        self.num_labels = len(self.dict_label_index)

        self.l1 = None
        self.linear = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loaded = False

        self.func_loss = None
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

    def load(self, is_eval=True):
        if not self.loaded:
            self.l1 = BertModel(BertConfig.from_pretrained(self.path_config_bert))
            self.classifier = torch.nn.Linear(self.classifier_input_size, self.classifier_output_size)
            self.load_state_dict(torch.load(self.path_file_model, map_location=torch.device(self.device)))

            print('Please Ignore warning message sent by BertTokenizer below')
            self.tokenizer = BertTokenizer.from_pretrained(self.path_folder_model)
            # self.factory_encoder = EncoderFactory(self.tokenizer, self.device, self.classifier_max_length)
             
            if is_eval:
                self.eval()
            else:
                self.func_loss = torch.nn.CrossEntropyLoss()
                self.train()
            # end
            self.loaded = True
        return self
# end

path_model = os.path.join('models','bert','target_v1')
classifier = SimpleBertClassifier(path_model)
classifier.load(is_eval=False)


path_train = os.path.join('data','test.csv')
path_label = os.path.join('data', 'label.json')

print('[{}] start main_train_and_evaluate with {} {}'.format(get_ts(), path_train, path_label))

max_length = classifier.classifier_max_length
output_dir = DIR_OUTPUT

(train_samples, valid_samples, train_labels, valid_labels), target_names = read_passages(path_train, path_label,
                                                                                            0.1)

tokenizer = classifier.tokenizer
train_encodings = tokenizer.batch_encode_plus(train_samples, truncation=True, padding=True, max_length=max_length,
                                                return_tensors='pt')
valid_encodings = tokenizer.batch_encode_plus(valid_samples, truncation=True, padding=True, max_length=max_length,
                                                return_tensors='pt')

train_dataset = SimpleDataset(train_encodings, train_labels)
valid_dataset = SimpleDataset(valid_encodings, valid_labels)

def compute_metrics(pred): # pred:  ['count', 'index', 'label_ids', 'predictions']
    labels = pred.label_ids.reshape(-1)
    preds = pred.predictions.argmax(-1).reshape(-1)

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
# end


training_args = TrainingArguments(
    output_dir=output_dir,  # output directory
    num_train_epochs=1,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=0,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    # load the best model when finished training (default metric is loss)    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    # logging_steps=1,  # log & save weights each logging_steps
    evaluation_strategy="epoch",  # evaluate each `logging_steps`
    learning_rate=2e-5,
    # save_strategy='epoch',
    metric_for_best_model='f1'
)

# trainer = Trainer(
#     model=model,  # the instantiated Transformers model to be trained
#     args=training_args,  # training arguments, defined above
#     train_dataset=train_dataset,  # training dataset
#     eval_dataset=valid_dataset,  # evaluation dataset
#     compute_metrics=compute_metrics,  # the callback that computes metrics of interest
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
# )

trainer = Trainer(
    model=classifier,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=valid_dataset,  # evaluation dataset
    compute_metrics=compute_metrics
)

print('[{}] start training...'.format(get_ts()))
trainer.train()

info_state_model = trainer.evaluate()
print('[{}] finish training.'.format(get_ts()))

################## start to do eval ##################
