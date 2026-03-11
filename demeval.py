import torchvision.models as models
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
import os
import glob
from PIL import Image
device="cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
i=31
folder=glob.glob(f"input/test/Moderate Dementia/OAS1_{i:04d}_MR1_mpr-1_*.jpg")
print
#l1=os.listdir(folder)
#l2=[os.path.join(folder,h) for h in l1]
images = [Image.open(p) for p in folder]
transform_image=[transform(img) for img in images]
x=torch.stack(transform_image)

o={0:'Moderate Dementia', 1:'Non Demented', 2:'Very mild Dementia'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
#model.classifier = nn.Linear(model.classifier.in_features, 3)

model = model.to(device)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 3).to(device)

checkpoint = torch.load("weights/model_checkpoint_22.pth", map_location=device)
model.load_state_dict(checkpoint["model"])

model.eval()
with torch.no_grad():
    
    x=x.to(device)
    pred=model(x)
  
    
    
    label=torch.mean(F.softmax(pred,dim=1),axis=0)
    
    print(o[label.argmax().item()])
