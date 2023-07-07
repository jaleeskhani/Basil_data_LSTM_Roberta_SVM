from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import SnowballStemmer
import re, torch, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = pd.read_csv('Bias_Data.csv')

drop_col_name = 'Lexical'
data = data[data.Label != drop_col_name]
data = data[~data.Line.isnull()]

data.Label, label_names  = data.Label.factorize()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
#         if token not in stop_words and token != 'im':
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)

# Line by line preprocess function calling
data.Line = data.Line.apply(lambda x: preprocess(x))


# To get vocabularies. This returns vector of unique tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.Line)

word_index = tokenizer.word_index
vocabulary_size = len(tokenizer.word_index) + 1
# print("Vocabulary : {}".format(word_index))

MAX_SEQUENCE_LENGTH = 20
lines = pad_sequences(tokenizer.texts_to_sequences(data.Line),
                        maxlen = MAX_SEQUENCE_LENGTH)

data.Line = lines.tolist()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(lines, data.Label, test_size=0.3)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
print('training: ')
#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred)
report_roberta = pd.DataFrame(classification_report(y_test, y_pred, target_names=label_names, output_dict=True)).transpose()
print('\nClassification Report:\n', report_roberta)
print('\nConfusion Matrix:\n', cm_svm)
