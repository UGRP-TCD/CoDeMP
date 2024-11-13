
import numpy as np
import fslic
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
from shadow import remove_shadow  


img = io.imread('horses.jpg')

# img = remove_shadow(img)

downscale_factor = 0.4
img_resized = resize(img, (int(img.shape[0] * downscale_factor), int(img.shape[1] * downscale_factor)), anti_aliasing=True)


img_lab = color.rgb2lab(img_resized)
h, w, d = img_lab.shape
img_flat = img_lab.reshape(-1).tolist()

clusters = 100
compactness = 100
result = np.array(fslic.fslic(img_flat, w, h, d, clusters, compactness, 10, 1, 1))
result = result.reshape(h, w, d)


result_rgb = color.lab2rgb(result)


plt.imshow(result_rgb)
plt.axis('off')
plt.show()

io.imsave('fslic_result.png', (result_rgb * 255).astype(np.uint8))