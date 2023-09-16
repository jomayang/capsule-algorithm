
import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image


from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet


# Build Colorization model

# Generator

def res34_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    Col_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return Col_G

class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)

# Discriminator

SIZE = 256
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# GAN loss

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

# Model Initialization

def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

# Putting everything together

class Colorizer_GAN(nn.Module):
    def __init__(self, Col_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if Col_G is None:
            self.Col_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.Col_G = Col_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.Col_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
    
    def setup_input_L(self, data):
        self.L = data['L'].to(self.device)

    def setup_input_L_img(self, data):
        self.L = data.to(self.device)

    def forward(self):
        self.fake_color = self.Col_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.Col_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

# Utility functions

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

#########################################################################################
######                             From here    v                                  ######
#########################################################################################

def image_mode_check(path):
  
  # Function returns the image mode. In our case it returns 'RGB' or 'L'. 
  # path: Input String, should be the image directory.

  return (Image.open(path)).mode

def maximum(a,b):
  return a if a >= b else b

# Main algorithm

#retrieve pathological sequences pointers indeces.

def retrieve_pathological_seq(path):

    paths = sorted(glob.glob(path + "/*.jpg")) 
    
    len_ = 0
    start = 0
    is_moving = False
    
    pathological_positions = []
    pathological_pointers = []
    
    for img_path in paths:
    
      if (image_mode_check(img_path)=="RGB"):
        
        if is_moving == False:
          start = paths.index(img_path)
          len_ = 1
          is_moving = True
        else:
          len_ = len_ + 1
    
      else:
        if is_moving:
          pathological_positions.append((start, len_))
          pathological_pointers.append((start, (start + (len_ - 1)) + 1))
          len_ = 0
          start = 0
          is_moving = False
        else:
          continue
    print("[Done] retrieve pathological sequences pointers indeces")

    return paths, pathological_pointers, pathological_positions

def colorize_emergency_seq(paths, pathological_pointers, len_emergency_seq, model):

    # case 1: one pathsological sequence has been found
    
    if len(pathological_pointers)<=1:
    
      print("One pathological sequence has been detected...")
      print("------------ Pathological sequence ------------")
      #visualize_rgb(paths[pathological_pointers[0][0]:pathological_pointers[0][1]])
      max_ = maximum(len(paths[:pathological_pointers[0][0]]), len(paths[pathological_pointers[0][1]:-1]))
    
      for x in range(max_):
    
        if x <= len(paths[:pathological_pointers[0][0]]): # To colorize prior sequence
          im_i = paths[(pathological_pointers[0][0] - x)]
          im_i = colorized_rgb(im_i, model)
          #visualize_colorized_image(im_i)
    
        if x <= len(paths[pathological_pointers[0][1]:-1]): # To colorize next sequence
          im_j = paths[(pathological_pointers[0][1] + x)]
          im_j = colorized_rgb(im_j, model)
          #visualize_colorized_image(im_j)
    
    # case 2: more than one pathsological sequences to colorize for emergency use
    else:
    
      print(f"{len(pathological_pointers)} pathological sequences has been detected...")
      
      for sequence in pathological_pointers:
        
        #visualize_rgb(paths[(sequence[0]+1):sequence[1]]) 
        print(f"Colorizing the {sequence} is on progress..")
    
        for x in range(len_emergency_seq):
    
          if x <= len_emergency_seq:  
            
            im_i = paths[(sequence[0] - x)]
            im_i = colorized_rgb(im_i, model)
            #visualize_colorized_image(im_i) 
    
            im_j = paths[(sequence[1] + x)]
            im_j = colorized_rgb(im_j, model)
            #visualize_colorized_image(im_j)
    
    return print("[Done] Colorization of all emergency sequence.")

def colorize_rest_of_img(paths, pathological_pointers, len_emergency_seq, model):

    # Colorization of the rest L images.

    if len(pathological_pointers) > 1:

        for y in range(len(pathological_pointers)):
        
            if pathological_pointers[0][0] == pathological_pointers[y][0]: # For first sequence 
        
              next_i = len(paths[:(pathological_pointers[0][0] - len_emergency_seq)])
              next_j = len(paths[(pathological_pointers[0][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])  
              
            elif pathological_pointers[-1][-1] == pathological_pointers[y][-1]: # For last sequences 
        
              next_i = 0
              next_j = len(paths[(pathological_pointers[-1][-1]+len_emergency_seq):-1])
        
            else: # For mid sequences 
        
              next_i = 0
              next_j = len(paths[(pathological_pointers[y][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])
        
            max_ = maximum(next_i, next_j)
        
            for x in range((max_+1)):
        
              if x <= next_i and next_i != 0: # To colorize prior sequence
        
                im_i = paths[(pathological_pointers[y][0]- len_emergency_seq - x)]
                im_i = colorized_rgb(im_i)
                visualize_colorized_image(im_i)
        
              if x <= next_j and next_j != 0: # To colorize next sequence
        
                im_j = paths[(pathological_pointers[y][1] + len_emergency_seq+ x)]
                im_j = colorized_rgb(im_j)
                visualize_colorized_image(im_j)
        print("[Done] Colorization of all grayscale images.")

    else:
        return None

    for y in range(len(pathological_pointers)):
    
        if pathological_pointers[0][0] == pathological_pointers[y][0]:
    
          next_i = len(paths[:(pathological_pointers[0][0] - len_emergency_seq)])
          next_j = len(paths[(pathological_pointers[0][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])  
          
        elif pathological_pointers[-1][-1] == pathological_pointers[y][-1]:
    
          next_i = 0
          next_j = len(paths[(pathological_pointers[-1][-1]+len_emergency_seq):-1])
    
        else: 
    
          next_i = 0
          next_j = len(paths[(pathological_pointers[y][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])
    
        max_ = maximum(next_i, next_j)
    
        for x in range((max_+1)):
    
          if x <= next_i and next_i != 0: 
    
            im_i = paths[(pathological_pointers[y][0]- len_emergency_seq - x)]
            im_i = colorized_rgb(im_i, model)
            #visualize_colorized_image(im_i)
    
          if x <= next_j and next_j != 0: 
    
            im_j = paths[(pathological_pointers[y][1] + len_emergency_seq+ x)]
            im_j = colorized_rgb(im_j, model)
            #visualize_colorized_image(im_j)
    return print("[Done] Colorization of all grayscale images.")