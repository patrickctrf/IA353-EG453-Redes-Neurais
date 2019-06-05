import tensorflow as tf
import os
import threading

myMutex = threading.Lock()
value = "teste"

numeroDeNeuronios = []
numeroDeEpocas = []
numeroDeCamadas = []
numeroDeDropout = []
taxaDeAcertos = []

# Vamos colocar uma thread para treinar cada rede com um numero especifico de camadas.
def threadPorCamadas(camadas):
	
	global value;
	
	# Os valores que utilizaremos para dropout variarao de 10% a 90% (instrucao abaixo).
	valoresDropout = range(10, 100, 50)# Variaremos de 10% em 10%.
	valoresDropout = [i/100 for i in valoresDropout]# Converte de porcentagem para escala de 0 a 1.
	
	# Testando resultados com diferentes quantidades de epocas.
	for epocas in [4,5]:#,6,7]:
	
		# Testando resultados com diferentes quantidades de neuronios.
		for neuronios in [100, 200]:#, 300, 400, 512]:
	
			# Este loop fica responsável por treinar com diferentes taxas de dropout.
			# "i" eh o valor a cada iteracao.
			for taxaDropout in valoresDropout:
				mnist = tf.keras.datasets.mnist
				(x_train, y_train),(x_test, y_test) = mnist.load_data()
				x_train, x_test = x_train / 255.0, x_test / 255.0
				model = tf.keras.models.Sequential([
				 tf.keras.layers.Flatten(),
				 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
				 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
				 tf.keras.layers.Dense(10, activation=tf.nn.softmax)
				])
				model.compile(optimizer='adam',
				 loss='sparse_categorical_crossentropy',
				 metrics=['accuracy'])
				model.fit(x_train, y_train, epochs=epocas)
				value = model.evaluate(x_test, y_test)
				model_json = model.to_json()
				json_file = open("model_MLP.json", "w")
				json_file.write(model_json)
				json_file.close()
				model.save_weights("model_MLP.h5")
				print("Model saved to disk")
				print(value)
				os.getcwd()
				
				myMutex.acquire()
				numeroDeNeuronios.append(neuronios)
				numeroDeEpocas.append(epocas)
				numeroDeCamadas.append(camadas)
				numeroDeDropout.append(taxaDropout)
				taxaDeAcertos.append(value)
				myMutex.release()
	
	
if __name__ == '__main__':

	
	camadas1 = threading.Thread(target=threadPorCamadas,args=(1,))
#	camadas2 = threading.Thread(target=threadPorCamadas,args=(2))
#	camadas3 = threading.Thread(target=threadPorCamadas,args=(3))
#	camadas4 = threading.Thread(target=threadPorCamadas,args=(4))
	
	camadas1.start()
#	camadas2.start()
#	camadas3.start()
#	camadas4.start()
