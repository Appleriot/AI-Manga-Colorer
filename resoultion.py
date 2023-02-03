import cv2
import os
import matplotlib.pyplot as plt
import random

n = random.randint(1, 999999)
names = "edit"

img = cv2.imread('edit8553682.jpeg')

sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = 'EDSR_x4.pb'

sr.readModel(path)

sr.setModel('edsr', 4)

result = sr.upsample(img)

resized = cv2.resize(result, dsize=None, fx=4, fy=4)

plt.figure(figsize=(12,8))
plt.imshow(result[:,:,::-1])
plt.imshow(resized[:,:,::-1])
plt.show()

if os.path.exists(names) == True:
    n = str(n)
    new_name = 'edit' + str(n)
    cv2.imwrite(f'upgrade/{new_name}.png', result)
    cv2.imwrite(f'upgrade/resized{new_name}.png', resized)
else:
    n = random.randint(1,9999999)
    n = str(n)
    new_name = 'edit' + n
    cv2.imwrite(f'upgrade/{new_name}.png', result)
    cv2.imwrite(f'upgrade/resized{new_name}.png', resized)

