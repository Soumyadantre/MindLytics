from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/mental_health_model.h5')

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["How often do you feel sad or hopeless?\nA) Almost every day\nB) Rarely\nC) Never",
                        "Do you have trouble concentrating on things like reading or watching TV?\nA) Yes\nB) No",
                        "How often do you have trouble sleeping?\nA) Almost every night\nB) Occasionally\nC) Never",
                        "Do you often feel worried or on edge?\nA) Yes\nB) No",
                        "Do you ever have thoughts of harming yourself?\nA) Yes\nB) No",
                        "How often do you feel excessively happy or 'high'?\nA) Almost every day\nB) Rarely\nC) Never",
                        "Do you frequently experience sudden and intense fear or panic?\nA) Yes\nB) No",
                        "How often do you experience intrusive thoughts or urges?\nA) Several times a day\nB) Occasionally\nC) Never",
                        "Do you frequently feel detached from reality or experience hallucinations?\nA) Yes\nB) No",
                        "How often do you have trouble controlling your temper?\nA) Almost every day\nB) Rarely\nC) Never",
                        "Do you often have trouble remembering things?\nA) Yes\nB) No",
                        "Do you feel like you have little energy or feel tired all the time?\nA) Yes\nB) No"])

# Routes
@app.route('/')
def home():
    questions = list(tokenizer.word_index.keys())
    return render_template('index.html', questions=questions)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_responses = []
        for question in tokenizer.word_index.keys():
            user_responses.append(request.form[question])
        
        # Tokenize and pad the user responses
        user_sequences = tokenizer.texts_to_sequences(user_responses)
        user_padded = pad_sequences(user_sequences, maxlen=20, padding='post', truncating='post')
        
        # Predict the mental health issue
        prediction = model.predict(user_padded)
        predicted_labels = np.argmax(prediction, axis=1)
        predicted_issues = [tokenizer.index_word[label] for label in predicted_labels]
        
        return render_template('result.html', predicted_issues=predicted_issues)

if __name__ == '__main__':
    app.run(debug=True)

