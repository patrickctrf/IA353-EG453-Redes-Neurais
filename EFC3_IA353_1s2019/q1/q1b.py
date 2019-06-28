import keras
from vis.visualization import visualize_activation
from vis.utils import utils
import matplotlib.pyplot as plot

model = keras.models.load_model('mnist_model.h5')
layer_idx = utils.find_layer_idx(model, 'nome_exclusivo_05')
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

for Lp in range(-9, 11, 1):
    for classe in range(0, 10): 
        plot.subplot(20,10,1+classe+10*(Lp+9))
        filter_idx = classe
        img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), verbose=True, max_iter=1000, tv_weight=1., lp_norm_weight=float(Lp/10.0))
        plot.imshow(img.squeeze(), cmap='seismic', interpolation='nearest')

# ajustando tamanho de exibição
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18, 18)
# fig.savefig('test2png.png', dpi=100, forward=True)
plot.show()
