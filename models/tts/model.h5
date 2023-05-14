import os
import numpy as np
import tensorflow as tf

class TTSModel:
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.sample_rate = 22050
        self.max_length = 100
        self.max_r = 5
        self.vocab_size = 148
        self.model = self._load_model()
        self.char_to_index = self._load_char_to_index()

    def _load_model(self):
        model = tf.keras.models.load_model(os.path.join(self.model_dir, "model.h5"))
        return model

    def _load_char_to_index(self):
        char_to_index = {}
        with open(os.path.join(self.model_dir, "char_to_index.txt"), "r", encoding="utf-8") as f:
            for line in f:
                char, index = line.strip().split("\t")
                char_to_index[char] = int(index)
        return char_to_index

    def _text_to_sequence(self, text):
        text = text.lower()
        sequence = []
        for c in text:
            if c in self.char_to_index:
                sequence.append(self.char_to_index[c])
            else:
                sequence.append(self.char_to_index[" "])
        sequence.append(self.char_to_index["~"])
        sequence = np.array(sequence)
        return sequence

    def _generate_melspectrogram(self, sequence):
        r = np.linspace(0, self.max_r, self.max_length * self.sample_rate)
        t = np.linspace(0, len(sequence) / self.sample_rate, len(sequence))
        melody = np.sin(2 * np.pi * sequence * r[:, np.newaxis])
        signal = np.zeros_like(melody[:, 0])
        for i in range(melody.shape[1]):
            signal += np.roll(melody[:, i], i * self.sample_rate)
        signal = np.minimum(np.maximum(signal, -1), 1)
        mel = tf.signal.stft(signal, frame_length=1024, frame_step=256, fft_length=1024)
        mel = tf.abs(mel)
        mel = tf.math.log(mel + 1e-6)
        return mel

    def synthesize(self, text):
        sequence = self._text_to_sequence(text)
        mel = self._generate_melspectrogram(sequence)
        mel = tf.expand_dims(mel, axis=0)
        audio = self.model(mel)
        audio = tf.squeeze(audio, axis=0)
        audio = np.array(audio)
        return audio
    def synthesize_to_file(self, text, filename):
        audio = self.synthesize(text)
        waveform = tf.audio.encode_wav(tf.expand_dims(audio, axis=-1), self.sample_rate)
        tf.io.write_file(filename, waveform)

    def synthesize_to_dir(self, texts, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        for i, text in enumerate(texts):
            filename = os.path.join(dir, f"{i}.wav")
            self.synthesize_to_file(text, filename)
        print(f"Synthesized {len(texts)} files to {dir}")

    def evaluate(self, test_dataset):
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss_sum = 0
        num_batches = 0
        for batch in test_dataset:
            input_seq, target_seq = batch
            input_seq = tf.cast(input_seq, tf.int32)
            target_seq = tf.cast(target_seq, tf.float32)
            mel_pred = self.model(input_seq)
            loss = loss_fn(target_seq, mel_pred)
            loss_sum += loss.numpy()
            num_batches += 1
        return loss_sum / num_batches

    def train(self, train_dataset, test_dataset, epochs, save_every=None):
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.MeanSquaredError()
        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
        for epoch in range(epochs):
            for batch in train_dataset:
                input_seq, target_seq = batch
                input_seq = tf.cast(input_seq, tf.int32)
                target_seq = tf.cast(target_seq, tf.float32)
                with tf.GradientTape() as tape:
                    mel_pred = self.model(input_seq)
                    loss = loss_fn(target_seq, mel_pred)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                train_loss_metric.update_state(loss)
            for batch in test_dataset:
                input_seq, target_seq = batch
                input_seq = tf.cast(input_seq, tf.int32)
                target_seq = tf.cast(target_seq, tf.float32)
                mel_pred = self.model(input_seq)
                loss = loss_fn(target_seq, mel_pred)
                test_loss_metric.update_state(loss)
            train_loss = train_loss_metric.result()
            test_loss = test_loss_metric.result()
            train_loss_metric.reset_states()
            test_loss_metric.reset_states()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}")
            if save_every and (epoch+1) % save_every == 0:
                self.model.save(os.path.join(self.model_dir, f"model_{epoch+1}.h5"))
