from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('mental_health_model.h5')

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["How often do you feel sad or hopeless?", 
                        "Do you have trouble concentrating on things like reading or watching TV?",
                        "How often do you have trouble sleeping?",
                        "Do you often feel worried or on edge?",
                        "Do you ever have thoughts of harming yourself?",
                        "How often do you feel excessively happy or 'high'?",
                        "Do you frequently experience sudden and intense fear or panic?",
                        "How often do you experience intrusive thoughts or urges?",
                        "Do you frequently feel detached from reality or experience hallucinations?",
                        "How often do you have trouble controlling your temper?",
                        "Do you often have trouble remembering things?",
                        "Do you feel like you have little energy or feel tired all the time?"])

# Routes
@app.route('/')
def home():
    questions = tokenizer.word_index.keys()
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
        predicted_label = np.argmax(prediction)
        predicted_issue = label_encoder.classes_[predicted_label]
        
        return render_template('result.html', predicted_issue=predicted_issue)

if __name__ == '__main__':
    app.run(debug=True)
