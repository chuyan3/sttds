export PYTHONPATH='/home/chuyan/deepspeech.pytorch'
rm -rf wav_tmp
mkdir wav_tmp
python deploy/server_socket.py
