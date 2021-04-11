import socket
import json 
import http.server
import socketserver
import http.server
import socketserver

class UDP_Server:
    def __init__(self):
        self.HOST = '127.0.0.1'
        self.PORT = 12345
        self.size = 10000
        self.data = []
        self.rcv = True

    def connect(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self.HOST, self.PORT))
            self.startReceive(s)
            

    def startReceive(self,s):
        while self.rcv:
            data,addr = s.recvfrom(self.size)
            data.decode('utf-8')
            if not data:
                break

    def stopReceive(self):
        self.rcv = False

    def writeToJSON(self,data):
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

 

server = UDP_Server()
server.connect()