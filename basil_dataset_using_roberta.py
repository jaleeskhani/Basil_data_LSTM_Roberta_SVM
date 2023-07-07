import torch, re, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW

from nltk.stem import SnowballStemmer

data = pd.read_csv('Bias_Data.csv')

drop_col_name = 'Lexical'
# data = data[data.Label != drop_col_name]
data = data[~data.Line.isnull()]

data.Label, label_names  = data.Label.factorize()
num_labels = label_names.shape[0]

stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)

# Line by line preprocess function calling
data.Line = data.Line.apply(lambda x: preprocess(x))

xdata = list(data.Line)
ydata = list(data.Label)

train_texts, test_texts, train_labels, test_labels = train_test_split(xdata, ydata, test_size=.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class data_to_Tesnor(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = data_to_Tesnor(train_encodings, train_labels)
test_dataset = data_to_Tesnor(test_encodings, test_labels)

batch_size=16
learn_rate = 5e-5
epochs = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = BertConfig.from_pretrained('bert-base-uncased', num_labels = num_labels)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

model.to(device)
model.train()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optim = AdamW(model.parameters(), lr=learn_rate)

tr_loss = []

for epoch in range(epochs):
    print('Epoch: ', epoch, '\n')
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        tr_loss.append(loss.data.tolist())
        loss.backward()
        optim.step()

test_loader = DataLoader(test_dataset, batch_size=batch_size)

pred_lab = []
model.eval()
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    pred_lab.extend(predictions.tolist())

cm_roberta = confusion_matrix(test_labels, pred_lab)
report_roberta = pd.DataFrame(classification_report(test_labels, pred_lab, target_names=label_names, output_dict=True)).transpose()
print('\nClassification Report:\n', report_roberta)
print('\nConfusion Matrix:\n', cm_roberta)