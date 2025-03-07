from input import InputHandle
from model import Network
import torch
from loguru import logger

class Classifier(InputHandle):
  def __init__(self, inputImage:str):
    super().__init__(inputImage)
    self.__model = Network()
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
        _, predicted_class = torch.max(output, 1)
        return f"Predicted Class :{predicted_class.item()}"  



if __name__ == "__main__":
  classifier = Classifier("./dogie.jpg")
  classifier.load_weights("BinaryClassifier.pth","cpu")
  prediction = classifier.predict()
  
  print(prediction)