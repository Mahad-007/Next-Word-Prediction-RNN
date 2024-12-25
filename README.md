Next-Word Prediction Using RNNs
This repository contains a project that implements a Next-Word Prediction Model using Recurrent Neural Networks (RNNs) in Python. The model is designed to predict the next word in a sentence based on its context, demonstrating key concepts in Natural Language Processing (NLP) and deep learning.

📌 Project Highlights
Built a Sequential RNN model with:
Embedding Layer: To represent words as dense vectors.
SimpleRNN Layer: For capturing sequential patterns.
Dense Output Layer: To predict the next word.
Preprocessed text data to generate input sequences for training.
Trained the model using categorical cross-entropy loss and the Adam optimizer.
🛠️ Technologies Used
Libraries: TensorFlow, Keras, NumPy
Model Layers: Embedding, SimpleRNN, Dense
Tools: Tokenizer, sequence padding, and categorical encoding
🚀 How It Works
Text Preprocessing:
Tokenized and padded input sentences to create training sequences.
Model Training:
Trained the RNN for 100 epochs with a vocabulary size of total_words.
Next-Word Prediction:
The model predicts the next word in a given sentence fragment.
Example Predictions:
Input: "My name is"
Prediction: "Mahad"
Input: "I love to"
Prediction: "play"
📂 Project Structure
bash
Copy code
├── next_word_prediction.py   # Main script for training and predictions
├── README.md                 # Project description
└── .gitignore                # Ignored files and folders
⚡ Getting Started
Clone the repository:
bash
Copy code
git clone https://github.com/<your-username>/Next-Word-Prediction-RNN.git
cd Next-Word-Prediction-RNN
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the script:
bash
Copy code
python next_word_prediction.py
🎯 Future Enhancements
Experiment with more advanced architectures like LSTMs or GRUs.
Use larger datasets for better predictions.
Add a web interface for real-time predictions.
