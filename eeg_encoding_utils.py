import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from cornet.cornet_s import CORnet_S
from torchvision import models

def create_model(device, dnn):
    if dnn == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, 4250) 
    if dnn == 'cornet_s':
        model = CORnet_S()
        model.decoder = nn.Sequential(
            model.decoder.avgpool,
            model.decoder.flatten,
            model.decoder.linear,
            nn.Linear(in_features=1000, out_features=4250), 
            model.decoder.output 
        )
    model = model.to(device)
    return model

def load_model_endocer(model_path, device):
    model = create_model(device, 'alexnet')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    model.eval()
    return model

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def generate_eeg(model, image_tensor, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        eeg_output = model(image_tensor).detach().cpu().numpy()
        eeg_output = np.reshape(eeg_output, (17, 250))
    return eeg_output

def save_eeg_signal(eeg_signal, save_dir, idx, category):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{category}_{idx + 1}.npy" 
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, eeg_signal)

def generate_and_save_eeg_for_all_images(model_path, test_image_list, save_dir, device, category_list):
    model = load_model_endocer(model_path, device)
    for idx, image_path in enumerate(test_image_list):
        image_tensor = preprocess_image(image_path, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        category = category_list[idx]
        save_eeg_signal(synthetic_eeg, save_dir, idx, category)

def generate_eeg_for_image(model_path, image_path, save_dir, device, category):
    model = load_model_endocer(model_path, device)
    image_tensor = preprocess_image(image_path, device)
    synthetic_eeg = generate_eeg(model, image_tensor, device)
    save_eeg_signal(synthetic_eeg, save_dir, 182, category)