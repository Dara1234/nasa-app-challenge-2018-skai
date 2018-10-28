import os
import ssl
from time import time
from urllib.request import urlopen
from tempfile import NamedTemporaryFile

# Set this appropriately.
# os.environ["THEANO_FLAGS"] = "device=cpu"
os.environ['KERAS_BACKEND'] = "theano"

import theano
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

from neural_style.utils import *
from neural_style.fast_neural_style.transformer_net import get_transformer_net

class Model:
    
    def __init__(self, model_path):
        self.X = theano.shared(np.array([[[[]]]], dtype=floatX))
        transformer_net = get_transformer_net(self.X, model_path)
        Xtr = transformer_net.output
        self.get_Xtr = theano.function([], Xtr)
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
    
    def style_image_by_path(self, path, size=None, output_path=None):
        img = load_and_preprocess_img(path, size=size)
        start = time()
        self.X.set_value(img)
        img_tr = self.get_Xtr()
        img_tr = img_tr[0, :, :, :]
        add_imagenet_mean(img_tr)
        img_tr = img_tr[::-1].transpose((1, 2, 0))
        img_tr = np.clip(img_tr, 0, 255).astype(np.uint8)
        time_taken = int((time() - start) * 1000)
        print("Took {}ms".format(time_taken))
        plt.figure(figsize=(10, 10))
        plt.imshow(img_tr)
        plt.axis("off")
        if output_path is not None:
            imsave(output_path, img_tr)
        self.X.set_value(np.array([[[[]]]], dtype=floatX))
    
    def style_image_by_url(self, url, size=None, output_path=None):
        with NamedTemporaryFile(buffering=0) as f:
            imdata = urlopen(url, context=self.ssl_context).read()
            f.write(imdata)
            f.flush()
            os.fsync(f)
            self.style_image_by_path(f.name, size, output_path)

model = Model("stained_glass.h5")