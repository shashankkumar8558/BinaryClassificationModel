from torchvision import transforms as T
import torch
from PIL import Image

class InputHandle():
  def __init__(self,inputImage:str)->None:
    self.inputImage_path = inputImage
    self.__compose = T.Compose([
      T.Resize((64,64)),
      T.ToTensor(),
      T.Normalize(mean=[0.75128635, 0.73187373, 0.66569826],std=[0.16277556, 0.16570601, 0.2318707])
    ])
    
  def process(self)->torch.Tensor:
    self.__image = Image.open(self.inputImage_path).convert('RGB')
    return self.__compose(self.__image)
    
    