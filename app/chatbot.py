import tensorflow as tf
import numpy as np

class ChatGPT:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/chatgpt/model.ckpt')
        with open('models/chatgpt/vocab.txt', 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f]
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}

    def generate_reply(self, message):
        # 对输入进行预处理，将文本转换为模型输入的数字序列
        input_tokens = message.split()
        input_ids = [self.token2idx.get(token, 0) for token in input_tokens]
        input_ids = [0] * (1024 - len(input_ids)) + input_ids[:1023] + [2]
        input_ids = np.array([input_ids])
        # 使用模型生成回复
        output_ids = self.model.predict(input_ids).argmax(axis=-1)
        # 将模型输出的数字序列转换为文本
        output_tokens = [self.idx2token.get(idx, '') for idx in output_ids]
        output = ''.join(output_tokens).replace(' ', '')
        # 返回生成的回复
        return output

