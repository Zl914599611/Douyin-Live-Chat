import websocket
import json

class DouYinWebSocket:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def on_message(self, ws, message):
        data = json.loads(message)
        if 'cmd' in data and data['cmd'] == 'DANMU_MSG':
            danmu = data['data']['content']
            for handler in self.handlers:
                handler.on_danmu(danmu)
        elif 'cmd' in data and data['cmd'] == 'GIFT_SEND':
            gift = {
                'name': data['data']['gift_name'],
                'count': data['data']['gift_num'],
                'user': data['data']['uname']
            }
            for handler in self.handlers:
                handler.on_gift(gift)
        elif 'cmd' in data and data['cmd'] == 'GUARD_JOIN':
            user = data['data']['username']
            for handler in self.handlers:
                handler.on_user_join(user)

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        print("websocket closed")

    def on_open(self, ws):
        print("websocket opened")

    def start(self):
        self.ws = websocket.WebSocketApp(self.url,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close,
                                         on_open=self.on_open)
        self.ws.run_forever()
