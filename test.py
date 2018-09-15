import configparser 
from predictor import Predictor

class Demo():
    def __init__(self,conf_path='configs/test.config'):
        config = configparser.ConfigParser()
        config.read(conf_path)
        if len(config.sections()) != 1:
            print("warning! read the first section in ",conf_test)
        section = config[config.sections()[0]]
        model_path = section.get("model_path")
        lm_path = section.get("lm_path")
        gpu = section.get("gpu")
        self.predictor = Predictor(model_path = model_path, lm_path = lm_path, gpu = gpu)
    def predict(self,audio_path):
        if audio_path.endswith("wav"):
            result = self.predictor.predict(audio_path)
            print(result)
        else:
            assert False
    
if __name__ == "__main__":
    demo = Demo()
    demo.predict("test.wav")

