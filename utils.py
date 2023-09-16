import torch
import numpy as np

import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from torchvision import transforms
from PIL import Image
import time

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    ''' Takes a batch of images. '''
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def get_norm_L(img_path):

  '''Function recieves grayscale "L" path, and returns a normalized image [Between -1 and 1] with a flot32 Tensor format. '''
  
  img_L = Image.open(img_path)
  img_L = np.array(img_L)
  img_L = img_L.astype("float32") # Converting L from uint8 to float32
  img_L = transforms.ToTensor()(img_L)
  
  L = img_L[[0], ...] / 255 * 100 # Between 0 and 100
  L = L[[0], ...] / 50. -1 # Between -1 and 1
  L = L.reshape(1, L.shape[0], L.shape[1], L.shape[2]) # Reshape to torch Tensor input shape.
  return L

def colorizer(img_L, model):
  '''
  Function recieves grayscale "L" image, and returns the RGB colorized image.
  '''
  model.Col_G.eval()
  with torch.no_grad():
    model.setup_input_L_img(img_L)
    model.forward()
  fake_color = model.fake_color.detach()
  L = model.L
  fake_imgs = lab_to_rgb(L, fake_color)
  return fake_imgs

def colorized_rgb(img_path, model):
  
  ''' Function recieves grayscale "L" path, and returns the RGB colorized image.'''
  
  L = get_norm_L(img_path)
  return colorizer(L, model)


def visualize_colorized_image (img, save=False):############ Function to modify ############

    ''' Function recieves a colorized image as a path(type of List), and returns a plot of an RGB image.'''
    
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(5*cm, 5*cm))
    plt.imshow(img[0])
    if save:
        fig.savefig(f"colorization_{time.time()}.jpg")


def visualize_rgb(sequence, save=False):############ Function to modify ############
  
  ''' Function recieves pathological RGB images as a list of paths, and returns a plot of all RGB images. '''
  
  cm = 1/2.54  # centimeters in inches
  FIGSIZE = 30 * cm
  for i in range(len(sequence)):
    img = Image.open(sequence[i])
    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
    ax = plt.subplot(1, len(sequence), i + 1)
    ax.imshow(img)
    ax.axis("off")
    
  plt.show()
  if save:
        fig.savefig(f"pathological_{time.time()}.jpg")

def image_mode_check(path):
  
  ''' Function returns the image mode. In our case it returns 'RGB' or 'L'. 
      path: Input String, should be the image directory. '''

  return (Image.open(path)).mode

def maximum(a,b):
  return a if a >= b else b