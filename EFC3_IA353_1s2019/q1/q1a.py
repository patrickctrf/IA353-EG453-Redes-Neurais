import keras
import innvestigate
import matplotlib.pyplot as plot
import tensorflow as tf

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.load_model('mnist_model.h5')
model_wo_sm = innvestigate.utils.keras.graph.model_wo_softmax(model)

# Cada indice do vetor x_test contem uma imagem. Aqui definimos quais acessaremos.
# Sei que defini mais imagens do que o requisitado, mas o vetor itneiro so sera usado caso se torne conveniente.
imagensParaAnalisar = [1001,3,1000,999,1003,2003,3000,3003,4000,4003,5000,5003]


# 'i' itera sobre as COLUNAS do subplot que estamos construindo.
# Cada coluna exibe as imagens de uma classe (um numero).
i=0;
while i<6: 
    imagem = x_test[imagensParaAnalisar[i]:imagensParaAnalisar[i]+1]
    
    eixo = plot.subplot(6,7,1+7*i)
    if i==0: eixo.set_title("Original") 
    plot.imshow(imagem.squeeze(), cmap='gray', interpolation='nearest')


    eixo = plot.subplot(6,7,2+7*i)
    analyzer = innvestigate.analyzer.Gradient(model=model_wo_sm)
    analysis = analyzer.analyze(imagem)
    if i==0: eixo.set_title("Gradient") 
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    
    eixo = plot.subplot(6,7,3+7*i)
    analyzer = innvestigate.analyzer.SmoothGrad(model=model_wo_sm)
    analysis = analyzer.analyze(imagem)
    if i==0: eixo.set_title("SmoothGrad") 
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    
    eixo = plot.subplot(6,7,4+7*i)
    analyzer = innvestigate.analyzer.DeepTaylor(model=model_wo_sm)
    analysis = analyzer.analyze(imagem)
    if i==0: eixo.set_title("DeepTaylor") 
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    
    eixo = plot.subplot(6,7,5+7*i)
    analyzer = innvestigate.analyzer.LRPAlphaBeta(model=model_wo_sm, alpha=2, beta=1)
    analysis = analyzer.analyze(imagem)
    if i==0: eixo.set_title("LRPAlphaBeta") 
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    
    eixo = plot.subplot(6,7,6+7*i)
    analyzer = innvestigate.analyzer.LRPEpsilon(model=model_wo_sm, epsilon=1)
    analysis = analyzer.analyze(imagem)
    if i==0: eixo.set_title("LRPEpsilon") 
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    
    eixo = plot.subplot(6,7,7+7*i)
    analyzer = innvestigate.analyzer.LRPZ(model=model_wo_sm)
    analysis = analyzer.analyze(imagem)
    if i==0: eixo.set_title("LRPZ") 
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    
    
    i = i + 1;
    
    
plot.show() 
