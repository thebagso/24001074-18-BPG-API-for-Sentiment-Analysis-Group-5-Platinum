import re
import pandas as pd
import pickle
import numpy as np
import sqlite3

from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from flask import Flask, jsonify, request
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from

#### flask
app = Flask(__name__)

########swagger
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info={
        'title': LazyString(lambda: 'Kelompok 5 DSC18 - Final Project'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host=LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

kamusalaydf = pd.read_csv('/Users/0011-21-pt.lbb/binar-dsc/binar-dsc/binar-dsc/coding_binar/Asset Challenge/new_kamusalay.csv', encoding='ISO-8859-1', header=None)
kamusalay_dict = dict(zip(kamusalaydf[0], kamusalaydf[1]))
def koreksi_alay(text):
    return ' '.join([kamusalay_dict[word] if word in kamusalay_dict else word for word in text.split(' ')])

#Extend cleansing
def cleansing(sent):
    if not sent:  # Check if sent is empty or None
        return ""  # Return empty string for empty input
        
    sent = re.sub(r'\\x[0-9a-fA-F]{2}', '', sent) #Menghapus unicode
    sent = re.sub(r'\d', '', sent) #Menghapus angka dari teks
    sent = re.sub(r'\\\\n|\\n', '', sent) #Menghapus kata \\n dan \n
    sent = re.sub(r'[^\w\s,.?!]', '', sent) #Menghapus karakter selain huruf, angka, spasi, koma, titik, tanda tanya, dan tanda seru
    sent = re.sub(r'\b(USER|RT|URL)\b', '', sent) #Menghapus kata "USER" 
    sent = re.sub(r'(\?{2,})', '?', sent) #Menghapus karakter "?" lebih dari 2 berturut-turut menjadi 1 saja
    sent = re.sub(r'(\,{2,})', '', sent) #Menghapus karakter "," lebih dari 2 berturut-turut menjadi 1 saja
    sent = re.sub(r'(\.{2,})', '.', sent) #Menghapus karakter "." lebih dari 2 berturut-turut menjadi 1 saja
    sent = re.sub(r'(\!{2,})', '!', sent) #Menghapus karakter "!" lebih dari 2 berturut-turut menjadi 1 saja
    sent = re.sub(r'[ð½]', '', sent) #Menghapus karakter "ð½"
    sent = re.sub(r'(?<=^)\.', '', sent) #Menghapus "." di kata awal
    sent = re.sub(r'\s{2,}', '', sent) #Menghapus spasi lebih dari 2
    sent = koreksi_alay(sent)
    sent = sent.lower()

    return sent

sentiment = ['negative', 'neutral', 'positive']

# Daftar file yang akan digunakan dalam API
h5_model = "/Users/0011-21-pt.lbb/binar-dsc/binar-dsc/binar-dsc/coding_binar/Challenge Submission/LSTM/lstm1.h5"
tokenizer_pickle = "/Users/0011-21-pt.lbb/binar-dsc/binar-dsc/binar-dsc/coding_binar/Challenge Submission/LSTM/tokenizer.pickle"
model_nn = "/Users/0011-21-pt.lbb/binar-dsc/binar-dsc/binar-dsc/coding_binar/Challenge Submission/Neural Network/model.p"
feature_nn = "/Users/0011-21-pt.lbb/binar-dsc/binar-dsc/binar-dsc/coding_binar/Challenge Submission/Neural Network/feature.p"

file = open(tokenizer_pickle, 'rb')
tokenizer_lstm = pickle.load(file)
file.close()

model_lstm = load_model(h5_model)

##body api
@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing', methods=['POST'])
def text_processing():
    input_text = None
    model = None

    # Check if input is JSON
    if request.is_json:
        data = request.get_json()
        input_text = data.get('text')
        model = data.get('model')
        print(f"Received JSON data: text={input_text}, model={model}")
    else:
        input_text = request.form.get('text')
        model = request.form.get('model')
        print(f"Received form data: text={input_text}, model={model}")

    if not input_text or not model:
        return jsonify({
            'text': input_text,
            'sentiment': None,
            'status_code': 400,
            'description': "Invalid input",
            'output': cleansing(input_text)
        }), 400

    text = [cleansing(input_text)]
    polarity = None  # Initialize polarity

    if model == 'lstm':
        predicted = tokenizer_lstm.texts_to_sequences(text)
        guess = pad_sequences(predicted, maxlen=77)

        prediction = model_lstm.predict(guess)
        polarity = np.argmax(prediction[0])
        
        print("Text: ", text[0])
        print("Sentiment: ", sentiment[polarity])
        json_response = {
        'text': input_text,
        'sentiment': sentiment[polarity],
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'output': cleansing(input_text)
    }    
    elif model == 'nn':
        with open(feature_nn, 'rb') as file:
            feature = pickle.load(file)

        with open(model_nn, 'rb') as file:
            new_model = pickle.load(file)

        # Feature Extraction
        transformed_text = feature.transform(text)
# Kita prediksi sentimennya
        result = new_model.predict(transformed_text)[0]
        if isinstance(result, int):
            polarity = result
        else:
            polarity = np.argmax(result)

        print("Text: ", text[0])
        print("Sentiment: ", result)
        json_response = {
        'text': input_text,
        'sentiment': result,
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'output': cleansing(input_text)
    }
    else:
        return jsonify({
            'text': input_text,
            'sentiment': None,
            'status_code': 400,
            'description': "Invalid model type",
            'output': cleansing(input_text)
        }), 400

    response_data = jsonify(json_response)
    return response_data

# CSV LSTM Model
@swag_from("docs/text_processing_file_lstm.yml", methods=['POST'])
@app.route('/text-processing-file-lstm', methods=['POST'])
def text_processing_file_lstm():
    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv to Pandas
    df = pd.read_csv(file, encoding='latin1')

    # Identify the column that contains the text data
    text_column = None
    for column in df.columns:
        if 'tweet' in column.lower():
            text_column = column
            break
    
    if not text_column:
        # If no suitable column is found, use the first column
        text_column = df.columns[0]

    # Extract texts to be processed as a list
    texts = df[text_column].replace(r'\\n', ' ', regex=True).to_list()

    # Clean the texts and predict sentiments
    cleaned_text = []
    label = []
    for input_text in texts:
        cleaned_text.append(input_text)
        text = [cleansing(input_text)]
        predicted = tokenizer_lstm.texts_to_sequences(text)
        guess = pad_sequences(predicted, maxlen=77)

        prediction = model_lstm.predict(guess)
        polarity = np.argmax(prediction[0])
        label.append(sentiment[polarity])

    cleaned_text = [cleansing(text) for text in texts]
    df_db = pd.DataFrame()
    df_db['Before_Cleaned'] = df[text_column]
    df_db['After_Cleaned'] = cleaned_text
    df_db['Sentiment'] = label

    conn = sqlite3.connect('tweet_sentiment_lstm.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tweet_table (
        Before_Cleaned TEXT,
        After_Cleaned TEXT,
        Sentiment TEXT
    )
    ''')
    for index, text in df_db.iterrows():
        cursor.execute('''
            INSERT INTO tweet_table VALUES (?,?,?)
        ''', tuple(text))

    conn.commit()
    conn.close()
    
    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': cleaned_text,
    }

    response_data = jsonify(json_response)
    return response_data

# CSV NN Model
@swag_from("docs/text_processing_file_nn.yml", methods=['POST'])
@app.route('/text-processing-file-nn', methods=['POST'])
def text_processing_file_nn():
    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv to Pandas
    df = pd.read_csv(file, encoding='latin1')

    # Identify the column that contains the text data
    text_column = None
    for column in df.columns:
        if 'tweet' in column.lower():
            text_column = column
            break
    
    if not text_column:
        # If no suitable column is found, use the first column
        text_column = df.columns[0]

    df[text_column] = df[text_column].replace(r'\\n', ' ', regex=True)
    texts = df[text_column].to_list()

    # Clean the texts and predict sentiments
    cleaned_text = []
    label = []
    for input_text in texts:
        file = open(feature_nn, 'rb')
        feature = pickle.load(file)
        file.close()

        file = open(model_nn, 'rb')
        new_model = pickle.load(file)
        file.close()

        # Feature Extraction
        text = feature.transform([cleansing(input_text)])
        
        # Predict sentiment
        result = new_model.predict(text)[0]
        polarity = result[0] if isinstance(result[0], int) else np.argmax(result[0])
        sentiment_label = result
        label.append(sentiment_label)

    cleaned_text = [cleansing(text) for text in texts]
    df_db = pd.DataFrame()
    df_db['Before_Cleaned'] = df[text_column]
    df_db['After_Cleaned'] = cleaned_text
    df_db['Sentiment'] = label

    conn = sqlite3.connect('tweet_sentiment_nn.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tweet_table (
        Before_Cleaned TEXT,
        After_Cleaned TEXT,
        Sentiment TEXT
    )
    ''')
    for index, text in df_db.iterrows():
        cursor.execute('''
            INSERT INTO tweet_table VALUES (?,?,?)
        ''', tuple(text))

    conn.commit()
    conn.close()

    json_response = {
        'data_text': cleaned_text,
        'data_sentiment': label,
        'status_code': 200,
        'description': "Teks yang sudah diproses",
    }

    response_data = jsonify(json_response)
    return response_data

##running api
if __name__ == '__main__':
    app.run()
