from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import SnowballStemmer
import re, pandas as pd, numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

data = pd.read_csv('bias_aug_data.csv')

drop_col_name = 'neutral'
data = data[data.Label != drop_col_name]
data = data[~data.Line.isnull()]

temp, label_names  = data.Label.factorize()

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

data.Line = data.Line.apply(lambda x: preprocess(x))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.Line)

word_index = tokenizer.word_index
vocabulary_size = len(tokenizer.word_index) + 1

MAX_SEQUENCE_LENGTH = 20
lines = pad_sequences(tokenizer.texts_to_sequences(data.Line),
                        maxlen = MAX_SEQUENCE_LENGTH)
labs = data.Label.values

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(np.array(labs))
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_labs = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(lines, onehot_encoded_labs, test_size=0.3)

num_class = label_names.shape[0]

batch_size=16
learn_rate = 0.01
epochs = 10

model = tf.keras.Sequential()

model.add(Input(shape=(None,)))
model.add(Embedding(input_dim=vocabulary_size, output_dim=200, trainable=True))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(num_class, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train,
                     epochs=epochs, batch_size=batch_size,
                     verbose=1,shuffle=True,validation_split=0.15)

predictions=model.predict(X_test)
y_test=np.argmax(y_test, axis=-1)
pred_y=np.argmax(predictions, axis=-1)

cm = confusion_matrix(y_test, pred_y)
report_roberta = pd.DataFrame(classification_report(y_test, pred_y, target_names=label_names, output_dict=True)).transpose()
print('\nClassification Report:\n', report_roberta)
print('\nConfusion Matrix:\n', cm)