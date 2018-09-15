import sys
import os
import os.path
import argparse
import time
from predictor import Predictor

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model-path', default='tts200000/final.pth',
                    help='Path to model file created by training')
parser.add_argument('--audio-path', default='01.wav',
                    help='Audio file to predict on')
parser.add_argument('--gpu', default=None, help='which gpu to use')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top-paths', default=3, type=int, help='number of beams to return')
beam_args.add_argument('--beam-width', default=20, type=int, help='Beam width to use')
beam_args.add_argument('--lm-path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff-top-n', default=20, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm-workers', default=1, type=int, help='Number of LM processes to use')
args = parser.parse_args()





def get_audio(audio_path):
    if os.path.isfile(audio_path):
        yield audio_path
    elif os.path.isdir(audio_path):
        for sub_file in os.listdir(audio_path):
            if sub_file.endswith('.wav'):
                yield audio_path+sub_file
    else:
        assert False



if __name__ == '__main__':
    total_time = 0
    
    predictor = Predictor(args.model_path, args.lm_path, args.gpu, args.beam_width, args.cutoff_top_n)
    audio_path = args.audio_path
    if audio_path.endswith('wav'):
        print('start')
        t = predictor.predict(audio_path)
        print(t)
    elif os.path.isdir(audio_path):
        for f in os.listdir(audio_path):
            if f.endswith('wav'):
                t = predictor.predict(audio_path+f)
                print(f)
                print(t)
                print()
            print('total time',total_time)
    elif audio_path.endswith('csv') or audio_path.endswith('txt'):
        with open(audio_path,'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if ',' in line:
                    wav,txt = line.split(',')
                    p = predictor.predict(wav)
                    with open(txt) as t:
                        label = t.read()
                        if label[-1] == '\n':
                            label = label[:-1]
                        print('label '+label)
                        print(args.model_path+' '+p)
        print("totaltime",total_time)

    else:
        print('warning:input can not understand')
