from input import InputHandle
from model import SimpleResNet
import torch
from loguru import logger
from torch.nn import functional as F
class Classifier(InputHandle):
  def __init__(self, inputImage:str):
    super().__init__(inputImage)
    self.__model = SimpleResNet(num_classes=1)
    self.__model.eval()
    
  def load_weights(self, model_state:str = None, device:str = 'cpu')->None:
    self.device = device
    self.__model.load_state_dict(
      torch.load(
        model_state,
        map_location=torch.device(device=device)
        ) )
    logger.info("Model Loaded Successfully")
    
  def predict(self)->str:
      image_tensor = self.process().unsqueeze(0)
      image_tensor = image_tensor.to(self.device)
      
      with torch.no_grad():
        
        output = self.__model(image_tensor)
        out = F.sigmoid(output)
        return "cat" if out <=0.5 else "dog" 



if __name__ == "__main__":
  classifier = Classifier("./dog.webp")
  classifier.load_weights("model.pth","cpu")
  prediction = classifier.predict()
  
  print(prediction)