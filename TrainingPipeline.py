import torch
import torchvision
import ssl
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tqdm.notebook as tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import os
# Custom helpers
from model import *
import nb_optimizers as opt

# Allow Download of CIFAR datasets
ssl._create_default_https_context = ssl._create_unverified_context

# Helpers

def create_init_models():
    seeds = [0,1024,2022]
    model_types = {"CIFAR_10":(3,10),"CIFAR_100":(3,100),"MNIST":(1,10),"Fashion_MNIST":(1,10)}
    if not os.path.isdir("init_models"):
        os.mkdir("init_models")
    for s in seeds:
        for name, (in_channel,class_nb) in model_types.items():
            torch.manual_seed(s)
            random.seed(s)
            np.random.seed(s)
            ### Model
            model = VGG(in_channel,class_nb)
            torch.save({"model_state_dict":model.state_dict()},"init_models/"+name+"_"+str(s)+".pth")

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def get_samplers(train_dataset,generator,shuffle=True,val_ratio=0.1):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_ratio * num_train))
    if shuffle:
        np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx,generator=generator)
    val_sampler = SubsetRandomSampler(val_idx,generator=generator)
    return train_sampler, val_sampler

def collect_weights(cnn_weights_list,linear_weights_list,model,channels_nb=3):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.in_channels == channels_nb:
                cnn_weights_list.append(m.weight.ravel().detach().cpu().numpy())
        elif isinstance(m, nn.Linear):
            linear_weights_list.append(m.weight.ravel().detach().cpu().numpy())

def train_step(model,train_dataloader,device,optimizer,criterion,epoch,cnn_layer_weights,linear_layer_weights):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    for inputs, targets in tqdm.tqdm(train_dataloader,leave=False):
        ### Collect Weights
        if (idx%4) == 0:
            collect_weights(cnn_layer_weights,linear_layer_weights,model)
        ### Perform training
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        ### Compute Accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        idx += 1
    print(f"At end of epoch {epoch} we have average loss {train_loss/total:.5f} and average accuracy {correct/total:.5f}%")
  
def validation_step(model,val_dataloader,device,criterion,best_acc,epoch,checkpoint_name="checkpoint"):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_dataloader,leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    # Save checkpoint.
    accuracy = 100.*correct/total
    if accuracy > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_name):
            os.mkdir(checkpoint_name)
        torch.save(state, "./"+checkpoint_name+"/ckpt.pth")
        print(f"New optimal model at epoch {epoch} saved with validation accuracy {correct/total:.5f}%")
    else:
        print(f"Validation accuracy {correct/total:.5f}%")
    return accuracy

def training_loop(max_epoch,dataloader_train,device,optimizer,criterion,model,
                  dataloader_val,ckpt_name,scheduler,cnn_layer_weights,linear_layer_weights):
  best_accuracy = -1
  for epoch in tqdm.tqdm(range(max_epoch)):
      train_step(model,dataloader_train,device,optimizer,criterion,epoch,cnn_layer_weights,linear_layer_weights)
      epoch_accuracy = validation_step(model,dataloader_val,device,criterion,best_accuracy,epoch,checkpoint_name=ckpt_name)
      if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
      if scheduler != None:
        scheduler.step()

def save_weights_for_viz(cnn_weights,linear_weights,basename):
    cnn_file = open(basename+"cnn_weights.npy","wb")
    linear_file = open(basename+"linear_weights.npy","wb")
    np.save(cnn_file,cnn_weights)
    np.save(linear_file,linear_weights)
    cnn_file.close()
    linear_file.close()

def training_pipeline(dataset,init_model_pth,optimizer_parameters,basename,seed=2022,batch_size=1024,max_epoch=75):
  ### Reproducibility
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  g = torch.Generator()
  g.manual_seed(seed)
  ### Download Datasets
  if dataset == "CIFAR10":
    dataset_train = torchvision.datasets.CIFAR10("data/",download=True)
    dataset_test = torchvision.datasets.CIFAR10("data/",download=True,train=False)
  elif dataset == "CIFAR100":
    dataset_train = torchvision.datasets.CIFAR100("data/",download=True)
    dataset_test = torchvision.datasets.CIFAR100("data/",download=True,train=False)
  elif dataset == "MNIST":
    dataset_train = torchvision.datasets.MNIST("data/",download=True)
    dataset_test = torchvision.datasets.MNIST("data/",download=True,train=False)
  elif dataset == "FashionMNIST":
    dataset_train = torchvision.datasets.FashionMNIST("data/",download=True)
    dataset_test = torchvision.datasets.FashionMNIST("data/",download=True,train=False)
  else:
    raise Exception("Unavailable dataset, please select among CIFAR10, CIFAR100, MNIST, FashionMNIST.")
  ### Compute initial Transform
  if dataset in ["CIFAR10","CIFAR100"]:
    mean_per_channel = tuple((dataset_train.data/255).mean(axis=(0,1,2)))
    std_per_channel = tuple((dataset_train.data/255).std(axis=(0,1,2)))
  else:
    mean_per_channel = (dataset_train.data.numpy()/255).mean()
    std_per_channel = (dataset_train.data.numpy()/255).std()
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean_per_channel, std_per_channel),
  ])
  ### Dataset Creation
  if dataset == "CIFAR10":
    dataset_train = torchvision.datasets.CIFAR10("data/",transform=transform)
    dataset_test = torchvision.datasets.CIFAR10("data/",transform=transform,train=False)
  elif dataset == "CIFAR100":
    dataset_train = torchvision.datasets.CIFAR100("data/",transform=transform)
    dataset_test = torchvision.datasets.CIFAR100("data/",transform=transform,train=False)
  elif dataset == "MNIST":
    dataset_train = torchvision.datasets.MNIST("data/",transform=transform)
    dataset_test = torchvision.datasets.MNIST("data/",transform=transform,train=False)
  elif dataset == "FashionMNIST":
    dataset_train = torchvision.datasets.FashionMNIST("data/",transform=transform)
    dataset_test = torchvision.datasets.FashionMNIST("data/",transform=transform,train=False)
  ### Validation Split
  train_sampler, val_sampler = get_samplers(dataset_train,g)
  ### Dataloaders creation
  dataloader_train = DataLoader(dataset_train,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, sampler=train_sampler,
                                      )
  dataloader_val = DataLoader(dataset_train,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, sampler=val_sampler,
                                      )
  dataloader_test = DataLoader(dataset_test,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, shuffle=True)
  ### Model Creation
  if dataset == "CIFAR10":
    in_c, out_c = (3,10)
  elif dataset == "CIFAR100":
    in_c, out_c = (3,100)
  elif dataset == "MNIST":
    in_c, out_c = (1,10)
  elif dataset == "FashionMNIST":
    in_c, out_c = (1,10)
  model = VGG(in_c,out_c)
  init_checkpoint = torch.load(init_model_pth)
  model.load_state_dict(init_checkpoint['model_state_dict'])
  model.to(device)
  if device.type == 'cuda':
      model = torch.nn.DataParallel(model)
      cudnn.benchmark = True
  ### Optimization
  criterion = nn.CrossEntropyLoss()
  optimizer = opt.createOptimizer(device, optimizer_parameters, model)
  scheduler = None
  ### Weights Collection
  cnn_layer_weights = []
  linear_layer_weights = []
  ### Saving names
  ckpt_name = basename+"val_ckpt"
  weight_folder_name = basename+"weights"
  ### Training Loop
  training_loop(max_epoch,dataloader_train,device,optimizer,criterion,model,
                dataloader_val,ckpt_name,scheduler,cnn_layer_weights,linear_layer_weights)
  ### Weights Saving
  if not os.path.isdir(weight_folder_name):
        os.mkdir(weight_folder_name)
  save_weights_for_viz(cnn_layer_weights,linear_layer_weights,weight_folder_name+"/")
  ### Save to Drive
  path_start_cnn = "/content/"+weight_folder_name+"/cnn_weights.npy"
  path_start_linear = "/content/"+weight_folder_name+"/linear_weights.npy"
  path_start_ckpt = "/content/"+ckpt_name+"/ckpt.pth"
  path_end_cnn = "/content/gdrive/MyDrive/opt_ml_project/saved_runs/cnn_weights_"+basename+".npy"
  path_end_linear = "/content/gdrive/MyDrive/opt_ml_project/saved_runs/linear_weights_"+basename+".npy"
  path_end_ckpt = "/content/gdrive/MyDrive/opt_ml_project/saved_runs/"+basename+"_ckpt.pth"
  return cnn_layer_weights,linear_layer_weights