import tensorflow as tf
import numpy as np
import os
import pygame

class TTS:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/tts/model.h5')

    def synthesize(self, text):
        # 对输入进行预处理，将文本转换为模型输入的数字序列
        input_ids = []
        for c in text:
            if c in self.token2idx:
                input_ids.append(self.token2idx[c])
        input_ids = np.array([input_ids])
        # 使用模型生成语音
        output = self.model.predict(input_ids)
        # 将语音写入临时文件
        temp_path = 'temp.wav'
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        tf.keras.models.save_model(self.model, 'temp.h5')
        os.system(f'python text2speech.py temp.h5 {temp_path}')
        # 播放语音
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()
        os.remove(temp_path)

