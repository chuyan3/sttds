import socket
from transcribe import Predictor
import numpy as np
import threading
import time
import wave
import math
import audioop
from queue import Queue


print('start predictor')
predictor = Predictor('/home/chuyan/deepspeech.pytorch/tts200000/final.pth','/home/chuyan/kenlm/build/batch2_4.arpa')
predictor.predict('/home/chuyan/cy_audio_data/wav/01.wav')
print('predictor ready')


_end = object()

class Receiver(threading.Thread):
    def __init__(self,queue):
        threading.Thread.__init__(self)
        self.queue = queue
        sk = socket.socket()
        sk.bind(("192.168.16.166",8080))
        sk.listen(5)
        self.cs,addr = sk.accept()
        print('linked to %s'% str(addr))


    def run(self):
        print('receiver start')
        datalen=0
        while True:
            received = self.cs.recv(1024)
            datalen += len(received)
            if not received:break
            self.queue.put(received)

        self.queue.put(_end)
        print('received data len',datalen)
        print('receiver ended')
        self.cs.close()

class Sender(threading.Thread):
    def __init__(self,queue):
        threading.Thread.__init__(self)
        sk = socket.socket()
        sk.bind(("192.168.16.166",20003))
        sk.listen(3)
        self.cs,addr2 = sk.accept()
        print('sk2 lined to %s'% str(addr2))
        self.queue = queue
        self.wav_data=[]
        self.text_all=['']

        self.sample_width = 2
        sample_rate = 16000
        chunk = 512
        pause_threshold = 0.8
        seconds_per_buffer = float(chunk)/sample_rate

        self.pause_buffer_count = int(math.ceil(pause_threshold/seconds_per_buffer))
        self.energy_threshold = 500 
        self.wav_name=0

    def predict_wave(self,wav_data):
        global predictor
        wav_name = 'wav_tmp/'+str(self.wav_name)+'.wav'
        wavefile=wave.open(wav_name,'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(16000)
        wavefile.writeframes(wav_data)
        wavefile.close()
        text = predictor.predict(wav_name) 
        self.wav_name += 1
        return text

    def process_current(self):
            text = self.predict_wave(bytes().join(self.wav_data))
            self.text_all[-1]=text

            print('//////////////////////')
            print('\n'.join(self.text_all))
            self.cs.send('\n'.join(self.text_all).encode('utf-8'))


    def run(self):
        print('sender start')

        pause_count = 0
        max_pause_count = -1
        contain_loud = False
        total_energy = float(0)
        while True:
            received = self.queue.get()
            if received is _end:
                break
            energy = audioop.rms(received,self.sample_width)
            if energy > self.energy_threshold:
                pause_count = 0
                contain_loud=True
            else:
                pause_count += 1

            if pause_count > self.pause_buffer_count:
                if not contain_loud:
                    self.wav_data.append(received)
                else:
                    print('another slice')
                    self.process_current()
                    self.text_all.append('')
                    self.wav_data = [received]     
                    contain_loud = False
                pause_count = 0
            else:
                self.wav_data.append(received)

        print('sender end')
        self.cs.close()
if __name__=='__main__':
    q = Queue()
    rv = Receiver(q)
    sd = Sender(q)
    rv.start()
    sd.start()
    rv.join()
    sd.join()
