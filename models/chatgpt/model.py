import tensorflow as tf
import numpy as np
import os

class ChatGPT:

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.max_length = 40
        self.temperature = 0.7
        self.padding_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unknown_token = "<UNKNOWN>"
        self.vocab_path = os.path.join(self.model_dir, "vocab.txt")
        self.vocab, self.word_to_index, self.index_to_word = self._load_vocab()
        self.model = self._load_model()

    def _load_vocab(self):
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f]
        word_to_index = {word: i for i, word in enumerate(vocab)}
        index_to_word = {i: word for i, word in enumerate(vocab)}
        return vocab, word_to_index, index_to_word

    def _load_model(self):
        model = tf.keras.models.load_model(os.path.join(self.model_dir, "model.h5"))
        return model

    def generate_response(self, input_text):
        input_seq = [self.word_to_index.get(word, self.word_to_index[self.unknown_token]) for word in input_text]
        input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=self.max_length, padding="post", value=self.word_to_index[self.padding_token])
        prediction = self.model(input_seq)[0]
        prediction = prediction / self.temperature
        prediction = tf.nn.softmax(prediction).numpy()
        predicted_index = np.random.choice(len(prediction), p=prediction)
        predicted_word = self.index_to_word.get(predicted_index, self.unknown_token)
        response = [predicted_word]
        while predicted_word != self.end_token and len(response) < self.max_length:
            input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq[0][1:].tolist() + [predicted_index]], maxlen=self.max_length, padding="post", value=self.word_to_index[self.padding_token])
            prediction = self.model(input_seq)[0]
            prediction = prediction / self.temperature
            prediction = tf.nn.softmax(prediction).numpy()
            predicted_index = np.random.choice(len(prediction), p=prediction)
            predicted_word = self.index_to_word.get(predicted_index, self.unknown_token)
            response.append(predicted_word)
        if self.end_token in response:
            response = response[:response.index(self.end_token)]
        response_text = "".join(response)
        return response_text
