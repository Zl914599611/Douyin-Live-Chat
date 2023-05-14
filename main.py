import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import asyncio
import websockets
import requests
import random
import time
import threading
from chatbot import Chatbot
from speech import TextToSpeech
import douyin
import traceback

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 读取配置信息
with open('config.json', 'r') as f:
    config = json.load(f)

# 加载 Chatbot 模型
chatbot = Chatbot(model_dir='models/chatgpt')
print('Chatbot 模型加载完成')

# 加载 TTS 模型
tts = TextToSpeech(model_path='models/tts/model.h5')
print('TTS 模型加载完成')

# 建立 WebSocket 连接
async def connect_websocket():
    uri = config['douyin']['websocket_url']
    async with websockets.connect(uri) as websocket:
        # 发送进入直播间消息
        await websocket.send(douyin.get_enter_room_msg(config['douyin']['room_id'], config['douyin']['device_id']))
        while True:
            try:
                data = await websocket.recv()
                data = json.loads(data)
                if 'cmd' not in data:
                    continue
                cmd = data['cmd']
                if cmd == 'DANMU_MSG': # 弹幕消息
                    danmu = data['content'][1]
                    reply = chatbot.generate_reply(danmu) # 生成回复
                    print(f'回复: {reply}')
                    if reply:
                        speech_file = f'speech/{int(time.time() * 1000)}.mp3'
                        tts.synthesize(reply, speech_file) # 合成语音
                        os.system(f'mpg123 -q {speech_file}') # 自动播放语音
                elif cmd == 'SEND_GIFT': # 礼物消息
                    gift_info = data['gift']
                    speech_file = f'speech/{int(time.time() * 1000)}.mp3'
                    tts.synthesize(f"{gift_info['uname']} 送了 {gift_info['giftName']}", speech_file) # 合成语音
                    os.system(f'mpg123 -q {speech_file}') # 自动播放语音
                elif cmd == 'WELCOME_GUARD': # 进入消息
                    user_info = data['data']
                    speech_file = f'speech/{int(time.time() * 1000)}.mp3'
                    tts.synthesize(f"{user_info['username']} 进入直播间", speech_file) # 合成语音
                    os.system(f'mpg123 -q {speech_file}') # 自动播放语音
            except:
                traceback.print_exc()
                print('WebSocket 连接异常')
                break

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    # 启动 WebSocket 连接
    loop = asyncio.get_event_loop()
    loop.run_until_complete(connect_websocket())

    # 启动 Flask 应用
    app.run(debug=True)
