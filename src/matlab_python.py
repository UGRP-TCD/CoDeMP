import matlab.engine
import numpy as np
from PIL import Image
import time

def superpixel(img, down_factor=1):
    eng = matlab.engine.start_matlab()
    eng.cd(r'C:/UGRP/CoDeMP-simple/src/superpixel', nargout=0)
    pic = np.array(eng.main(img))
    return pic


if __name__ == "__main__":
    img_path = 'C:/UGRP/CoDeMP-simple/src/superpixel/12003.jpg'
    img = Image.open(img_path)
    pic = superpixel(img)
    image = Image.fromarray(pic)
    image.show()