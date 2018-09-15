import torch
from data_loader import SpectrogramParser
from decoder import GreedyDecoder
from model import DeepSpeech
import os

def decode_results(model, decoded_output):
    results = {"output":[]}
    for b in range(len(decoded_output)):
        for pi in range(len(decoded_output[b])):
            result = {'transcription': decoded_output[b][pi]}
            results['output'].append(result)
    return results


class Predictor(object):
    def __init__(self, model_path, lm_path = None, gpu = None, beam_width = 10, cutoff_top_n = 31, alpha = 0.8, beta = 1, lm_workers = 1):
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        torch.set_grad_enabled(False)
        self.model_path = model_path
        self.lm_path = lm_path
        if gpu:
            self.model = DeepSpeech.load_model(model_path, True)
        else:
            self.model = DeepSpeech.load_model(model_path, False)
            
        self.model.eval()
        labels = DeepSpeech.get_labels(self.model)
        audio_conf = DeepSpeech.get_audio_conf(self.model)

        if lm_path is not None:
            from decoder import BeamCTCDecoder
            self.decoder = BeamCTCDecoder(labels, lm_path=lm_path, alpha=alpha, beta=beta,
                                 cutoff_top_n=cutoff_top_n, cutoff_prob=1.0,
                                 beam_width=beam_width, num_processes=lm_workers)
        else:
            self.decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
        self.parser = SpectrogramParser(audio_conf, normalize=True)

    def predict(self, audio_path):
        spect = self.parser.parse_audio(audio_path).contiguous()
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        out = self.model(spect)
        decoded_output, _ = self.decoder.decode(out.data)
        transcriptions = decode_results(self.model, decoded_output)['output'][0]['transcription']
        return transcriptions

