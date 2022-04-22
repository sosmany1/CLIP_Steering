#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import pickle
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.append(os.path.abspath(os.getcwd()))
#sys.path.append(os.path.abspath('stylegan2-ada-pytorch'))
sys.path.append(os.path.abspath('/n/home05/sosmany/hays_lab/Lab/stylegan2-ada-pytorch'))
sys.path.append(os.path.abspath('/n/home05/sosmany/hays_lab/Lab/stylegan2-ada-pytorch/CLIP'))


#from clip_steering.clip_classifier_utils import SimpleTokenizer
from clip_classifier_utils import SimpleTokenizer

import numpy as np
from PIL import Image


# In[2]:


torch.cuda.current_device()


# In[4]:


get_ipython().system('pip install requests')


# In[6]:


with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module


# In[5]:


import matplotlib.pyplot as plt
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]

img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img_np = img.detach().cpu().numpy().squeeze()
print(img_np.min(), img_np.max())
img_np.shape


# In[4]:


get_ipython().system(' pip install requests')


# In[5]:


plt.imshow(img_np)


# In[9]:


# Set up clip classifier

#clip_model_path = 'pretrained/clip_ViT-B-32.pt'
#clip_model_path = '/n/home05/sosmany/hays_lab/Lab/CLIP_Steering/projects/styleGAN2/CLIP/notebooks/model.pt'
clip_model_path = '/root/.cache/clip/ViT-B-32.pt'



model = torch.jit.load(clip_model_path).cuda().eval()
input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


# In[10]:


# Image preprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()


# In[11]:


# Building features for clip
attributes = ["an evil face", "a radiant face", "a criminal face", "a beautiful face", "a handsome face", "a smart face"]
tokenizer = SimpleTokenizer()
text_tokens = [tokenizer.encode("This person has " + desc) for desc in attributes]


# In[ ]:


text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
sot_token = tokenizer.encoder['<|startoftext|>']
eot_token = tokenizer.encoder['<|endoftext|>']

for i, tokens in enumerate(text_tokens):
    tokens = [sot_token] + tokens + [eot_token]
    text_input[i, :len(tokens)] = torch.tensor(tokens)

text_input = text_input.cuda()


# In[ ]:


import pandas as pd
nsamples = 1000

relevant_stats = {
    'latent_z': [],
    'similarity': [],
    'top_bottom_diff': [],
    'prediction': []
}

for snum in range(nsamples):
    z = torch.randn([1, G.z_dim]).cuda()    # latent codes
    c = None  
    img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_np = img.detach().cpu().numpy().squeeze()

    image_input = preprocess(Image.fromarray(img_np.astype(np.uint8))).to('cuda')
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    with torch.no_grad():
        image_features = model.encode_image(image_input.unsqueeze(0)).float()
        text_features = model.encode_text(text_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    top_bottom_diff = similarity.max() - similarity.min()
    prediction = attributes[np.argmax(similarity)]
    
    if top_bottom_diff > 0.02:
        relevant_stats['latent_z'].append(z.detach().cpu().numpy())
        relevant_stats['similarity'].append(similarity)
        relevant_stats['top_bottom_diff'].append(top_bottom_diff)
        relevant_stats['prediction'].append(prediction)
        
    if snum % 200 == 0:
        print(f'Done: {snum} / {nsamples}')

df = pd.DataFrame(relevant_stats)


# In[ ]:


len(df)


# In[ ]:


sorted_df = df.sort_values(by='top_bottom_diff', ascending=False)


# In[3]:


idx = 17
z = torch.Tensor(sorted_df.iloc[idx]['latent_z']).cuda()    # latent codes
similarity = sorted_df.iloc[idx]['similarity']
prediction = sorted_df.iloc[idx]['prediction']
c = None  
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img_np = img.detach().cpu().numpy().squeeze()

plt.imshow(img_np)
print(attributes)
print(similarity)
print(prediction)


# In[ ]:


df.prediction.value_counts()


# In[ ]:


sorted_df[:20]


# In[ ]:





# In[ ]:





# In[ ]:




