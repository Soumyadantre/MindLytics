import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Example dataset (replace with your own dataset)
# Full dataset with multiple-choice questions and answers
dataset = [
    {"question": "How often do you feel sad or hopeless?\nA) Almost every day\nB) Rarely\nC) Never", "answer": "A", "mental_issue": "Depression"},
    {"question": "Do you have trouble concentrating on things like reading or watching TV?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Anxiety"},
    {"question": "How often do you have trouble sleeping?\nA) Almost every night\nB) Occasionally\nC) Never", "answer": "A", "mental_issue": "Insomnia"},
    {"question": "Do you often feel worried or on edge?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Anxiety"},
    {"question": "Do you ever have thoughts of harming yourself?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Depression"},
    {"question": "How often do you feel excessively happy or 'high'?\nA) Almost every day\nB) Rarely\nC) Never", "answer": "A", "mental_issue": "Bipolar Disorder"},
    {"question": "Do you frequently experience sudden and intense fear or panic?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Panic Disorder"},
    {"question": "How often do you experience intrusive thoughts or urges?\nA) Several times a day\nB) Occasionally\nC) Never", "answer": "A", "mental_issue": "Obsessive-Compulsive Disorder"},
    {"question": "Do you frequently feel detached from reality or experience hallucinations?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Schizophrenia"},
    {"question": "How often do you have trouble controlling your temper?\nA) Almost every day\nB) Rarely\nC) Never", "answer": "A", "mental_issue": "Intermittent Explosive Disorder"},
    {"question": "Do you often have trouble remembering things?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Memory Problems"},
    {"question": "Do you feel like you have little energy or feel tired all the time?\nA) Yes\nB) No", "answer": "A", "mental_issue": "Fatigue"}
]


# Extract questions, answers, and mental health issues from the dataset
questions = [entry['question'] for entry in dataset]
answers = [entry['answer'] for entry in dataset]
issues = [entry['mental_issue'] for entry in dataset]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

# Convert text data to sequences
question_sequences = tokenizer.texts_to_sequences(questions)

# Pad sequences to ensure uniform length
max_seq_length = max(len(seq) for seq in question_sequences)
question_padded = pad_sequences(question_sequences, maxlen=max_seq_length, padding='post', truncating='post')

# Encode answers as integers (A: 0, B: 1, C: 2)
answer_dict = {'A': 0, 'B': 1, 'C': 2}
answer_encoded = np.array([answer_dict[ans] for ans in answers])

# Define the RNN model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_seq_length),
    SimpleRNN(units=32),
    Dense(units=len(set(issues)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(question_padded, answer_encoded, epochs=10, batch_size=32)

# Save the trained model to an HDF5 file
model.save('trained_model/mental_health_model.h5')
