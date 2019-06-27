import keras
from vis.visualization import visualize_activation
from vis.utils import utils
import matplotlib.pyplot as plot

model = keras.models.load_model('mnist_model.h5')
layer_idx = utils.find_layer_idx(model, 'nome_exclusivo_05')
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

for Lp in range(0, 10):
    for classe in range(0, 10): 
        plot.subplot(10,10,1+classe+10*Lp)
        filter_idx = classe
        img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), verbose=True, max_iter=1000, tv_weight=1., lp_norm_weight=float(Lp))
        plot.imshow(img.squeeze(), cmap='seismic', interpolation='nearest')

plot.show()
