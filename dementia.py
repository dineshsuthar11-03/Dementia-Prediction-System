from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter
from collections import defaultdict
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
device="cuda" if  torch.cuda.is_available() else "cpu"
print(device)
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(device)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),


    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path
train_dataset = ImageFolderWithPaths(
    root="input/train",
    transform=transform
)
labels = [label for _, label,_ in train_dataset]
print("Train distribution:", Counter(labels))
test_dataset = ImageFolderWithPaths(
    root="input/test",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

for params in model.parameters():
    params.requires_grad=False
for param in model.features.denseblock4.parameters():
    param.requires_grad = True
for name, param in model.named_parameters():
    if param.requires_grad==True:
       print(name, param.requires_grad)

num_classes=3 
model.classifier = nn.Linear(model.classifier.in_features, num_classes).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

loss_fn=torch.nn.CrossEntropyLoss()

from tqdm import trange
import tqdm
epochs=30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for epoch in trange(epochs,desc="training",unit="step"):
 
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    epochs1=[]
    train_loss1=[]
    lr1=[]
    tr_acc=[]
    
    for x, y,paths in train_loader:
        
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        preds = model(x)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()
        preds1=torch.argmax(torch.nn.functional.softmax(preds,dim=1),dim=1)
        train_correct+=(preds1==y).sum().item()

        train_loss += loss.item()
        train_total+=y.size(0)
    train_loss /= len(train_loader)
    train_acc=100*train_correct/train_total
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    epochs1.append(epoch)
    lr1.append(current_lr)
    train_loss1.append(train_loss)
    tr_acc.append(train_acc)
        

        
    
   
     # ---- VALIDATION ----
    model.eval()
    val_loss = 0
    all_probs = []
    all_labels = []
    val_correct=0
    val_total=0
    val_loss1=[]
    auc1=[]
    p_auc=[]
    v_acc=[]
    patient_probs = defaultdict(list)
    patient_labels = {}
    with torch.no_grad():
        for x, y ,paths in test_loader:
            x = x.to(device)
            y = y.to(device)
           

            
            logits = model(x)
            probs=torch.nn.functional.softmax(logits,dim=1)
            probs1=torch.argmax(probs,dim=1)
            val_correct+=(probs1==y).sum().item()
            val_total+=y.size(0)
            for i in range(len(paths)):
                filename = paths[i]
                
                # extract patient id
                patient_id = filename.split("_")[1]  # '0052'

                patient_probs[patient_id].append(
                    probs[i].cpu().numpy()
                )

                patient_labels[patient_id] = y[i].cpu().item()
                final_probs = []
                final_labels = []

                
            val_loss += loss_fn(logits, y).item()
            #val_loss += val_loss.item()
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
    val_loss/=len(test_loader)
    val_loss1.append(val_loss)
    
    for patient_id in patient_probs:

                    avg_prob = np.mean(patient_probs[patient_id], axis=0)

                    final_probs.append(avg_prob)
                    final_labels.append(patient_labels[patient_id])

    final_probs = np.array(final_probs)
    final_labels = np.array(final_labels)
    val_acc=100*(val_correct/val_total)
    v_acc.append(val_acc)
        

            

            

    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    

    #print("Labels shape:", np.array(all_labels).shape)
    #print("Probs shape:", np.array(all_probs).shape)
    #print("Unique labels:", np.unique(all_labels))
    
    auc = roc_auc_score(
    all_labels,
    all_probs,
    multi_class="ovr")
    auc1.append(auc)
    #labels=[0,1,2])
    #print("auc: ",auc)
    patient_auc = roc_auc_score(
    final_labels,
    final_probs,
    multi_class="ovr")
    p_auc.append(patient_auc)     
    

    

    # ---------- LOG ----------
    
    
    print(
            f"Epoch [{epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"train acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val acc: {val_acc:.4f}"
            f"patient level auroc: {patient_auc:.4f}  "
            f"AUROC: {auc:.4f}"
        )
    
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": epoch,
    }, f"model_checkpoint_{epoch}.pth")

torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": epoch,
    }, f"model_checkpoint_final.pth")
print("model saved")



