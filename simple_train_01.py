import json
import os
import sys
from datetime import datetime
import pandas as pd
import torch
from torch import BoolTensor
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, EarlyStoppingCallback
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
    df = pd.read_csv(path_data)[:200]

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



path_train = os.path.join('.','test.csv')
path_label = os.path.join('.', 'label.json')

print('[{}] start main_train_and_evaluate with {} {}'.format(get_ts(), path_train, path_label))

model_name = MODEL_NAME
max_length = MAX_LENGTH
output_dir = DIR_OUTPUT

(train_samples, valid_samples, train_labels, valid_labels), target_names = read_passages(path_train, path_label,
                                                                                            0.1)

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
train_encodings = tokenizer.batch_encode_plus(train_samples, truncation=True, padding=True, max_length=max_length,
                                                return_tensors='pt')
valid_encodings = tokenizer.batch_encode_plus(valid_samples, truncation=True, padding=True, max_length=max_length,
                                                return_tensors='pt')

train_dataset = SimpleDataset(train_encodings, train_labels)
valid_dataset = SimpleDataset(valid_encodings, valid_labels)

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))

def compute_metrics(pred): # pred:  ['count', 'index', 'label_ids', 'predictions']
    labels = pred.label_ids.reshape(-1)
    preds = pred.predictions.argmax(-1).reshape(-1)

    print('jinyuj: len of labels: {}'.format(len(labels)))
    print('jinyuj: len of labels: {}'.format(len(preds)))

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
    model=model,  # the instantiated Transformers model to be trained
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


