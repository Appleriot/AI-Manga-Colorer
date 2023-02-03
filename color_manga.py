import os
import glob
import time
import random
import numpy
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from skimage.color import lab2rgb,rgb2lab
import cv2
from matplotlib import cm
import torch
from torchvision.utils import save_image
from damn import MainModel
from damn import build_res_unet
from damn import lab_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("final_models_weights_v23.pt", map_location=device))
model = MainModel(net_G=net_G)
model.eval()
path = "black_white_panels/0022-016.jpg"
img = PIL.Image.open(path)

img = img.resize((256, 256))
# to make it between -1 and 1
img = transforms.ToTensor()(img)[:1] * 2. - 1.
model.eval()
with torch.no_grad():
  preds = model.net_G(img.unsqueeze(0).to(device))
colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
cv2.imshow("y", colorized)
plt.subplot(1, 1, 1)
plt.imshow(colorized,interpolation='nearest', )
plt.axis('off')
plt.show()
n = random.randint(1, 999999)
names = "edit"
if os.path.exists(names) == True:
    n = str(n)
    new_name = 'edit' + str(n)
    im = Image.fromarray((colorized * 255).astype(numpy.uint8))
    cv2.imwrite(f"{new_name}.png")

else:
    n = random.randint(1,9999999)
    n = str(n)
    new_name = 'edit' + n
    im = Image.fromarray((colorized * 255).astype(numpy.uint8))
    im.save(f"{new_name}.jpeg")
