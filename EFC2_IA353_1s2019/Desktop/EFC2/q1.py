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
def thread1Camadas(camadas):
	
	# Para tirar a media das iteracoes, somaremos todas aqui e dividiremos pelo total.
	somaDasEficienciasDeCadaIteracao = 0
	
	# Os valores que utilizaremos para dropout variarao de 10% a 90% (instrucao abaixo).
	valoresDropout = range(10, 60, 10)# Variaremos de 10% em 10%.
	valoresDropout = [i/100 for i in valoresDropout]# Converte de porcentagem para escala de 0 a 1.
	
	# Testando resultados com diferentes quantidades de epocas.
	for epocas in [4, 8]:
	
		# Testando resultados com diferentes quantidades de neuronios.
		for neuronios in [256, 512]:
		
			# So para indicar em que passo da execucao estamos.
			print("\n\nepocas: " + str(epocas) + "\nCAMADAS" + str(camadas) + ": " + str(neuronios) + "\n\n")
	
			# Este loop fica responsável por treinar com diferentes taxas de dropout.
			# "i" eh o valor a cada iteracao.
			for taxaDropout in valoresDropout:
			
				# Repetimos o treinamento algumas vezes para tirar uma media da eficiencia
				for iteracaoMedia in range(1,4):
					mnist = tf.keras.datasets.mnist
					(x_train, y_train),(x_test, y_test) = mnist.load_data()
					x_train, x_test = x_train / 255.0, x_test / 255.0
					model = tf.keras.models.Sequential([
					 tf.keras.layers.Flatten(),
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(10, activation=tf.nn.softmax)
					])
					model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
					 loss='sparse_categorical_crossentropy',
					 metrics=['accuracy'])
					model.fit(x_train, y_train, epochs=epocas)
					value = model.evaluate(x_test, y_test)
					model_json = model.to_json()
					json_file = open("model_MLP1.json", "w")
					json_file.write(model_json)
					json_file.close()
					model.save_weights("model_MLP1.h5")
					print("Model saved to disk")
					os.getcwd()
					
					somaDasEficienciasDeCadaIteracao = value[1] + somaDasEficienciasDeCadaIteracao
					
				
				
				myMutex.acquire()
				numeroDeNeuronios.append(neuronios)
				numeroDeEpocas.append(epocas)
				numeroDeCamadas.append(camadas)
				numeroDeDropout.append(taxaDropout)
				taxaDeAcertos.append(somaDasEficienciasDeCadaIteracao/iteracaoMedia)
				myMutex.release()
				
				# Reiniciamos a soma.
				somaDasEficienciasDeCadaIteracao = 0
				
# Vamos colocar uma thread para treinar cada rede com um numero especifico de camadas.
def thread2Camadas(camadas):
	
	# Para tirar a media das iteracoes, somaremos todas aqui e dividiremos pelo total.
	somaDasEficienciasDeCadaIteracao = 0
	
	# Os valores que utilizaremos para dropout variarao de 10% a 90% (instrucao abaixo).
	valoresDropout = range(10, 60, 10)# Variaremos de 10% em 10%.
	valoresDropout = [i/100 for i in valoresDropout]# Converte de porcentagem para escala de 0 a 1.
	
	# Testando resultados com diferentes quantidades de epocas.
	for epocas in [4, 8]:
	
		# Testando resultados com diferentes quantidades de neuronios.
		for neuronios in [256, 512]:
		
			# So para indicar em que passo da execucao estamos.
			print("\n\nepocas: " + str(epocas) + "\nCAMADAS" + str(camadas) + ": " + str(neuronios) + "\n\n")
	
			# Este loop fica responsável por treinar com diferentes taxas de dropout.
			# "i" eh o valor a cada iteracao.
			for taxaDropout in valoresDropout:
			
				# Repetimos o treinamento algumas vezes para tirar uma media da eficiencia
				for iteracaoMedia in range(1,4):
					mnist = tf.keras.datasets.mnist
					(x_train, y_train),(x_test, y_test) = mnist.load_data()
					x_train, x_test = x_train / 255.0, x_test / 255.0
					model = tf.keras.models.Sequential([
					 tf.keras.layers.Flatten(),
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(10, activation=tf.nn.softmax)
					])
					model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
					 loss='sparse_categorical_crossentropy',
					 metrics=['accuracy'])
					model.fit(x_train, y_train, epochs=epocas)
					value = model.evaluate(x_test, y_test)
					model_json = model.to_json()
					json_file = open("model_MLP2.json", "w")
					json_file.write(model_json)
					json_file.close()
					model.save_weights("model_MLP2.h5")
					print("Model saved to disk")
					os.getcwd()
					
					somaDasEficienciasDeCadaIteracao = value[1] + somaDasEficienciasDeCadaIteracao
					
				
				
				myMutex.acquire()
				numeroDeNeuronios.append(neuronios)
				numeroDeEpocas.append(epocas)
				numeroDeCamadas.append(camadas)
				numeroDeDropout.append(taxaDropout)
				taxaDeAcertos.append(somaDasEficienciasDeCadaIteracao/iteracaoMedia)
				myMutex.release()
				
				# Reiniciamos a soma.
				somaDasEficienciasDeCadaIteracao = 0
				
# Vamos colocar uma thread para treinar cada rede com um numero especifico de camadas.
def thread3Camadas(camadas):
	
	# Para tirar a media das iteracoes, somaremos todas aqui e dividiremos pelo total.
	somaDasEficienciasDeCadaIteracao = 0
	
	# Os valores que utilizaremos para dropout variarao de 10% a 90% (instrucao abaixo).
	valoresDropout = range(10, 60, 10)# Variaremos de 10% em 10%.
	valoresDropout = [i/100 for i in valoresDropout]# Converte de porcentagem para escala de 0 a 1.
	
	# Testando resultados com diferentes quantidades de epocas.
	for epocas in [4, 8]:
	
		# Testando resultados com diferentes quantidades de neuronios.
		for neuronios in [256, 512]:
		
			# So para indicar em que passo da execucao estamos.
			print("\n\nepocas: " + str(epocas) + "\nCAMADAS" + str(camadas) + ": " + str(neuronios) + "\n\n")
	
			# Este loop fica responsável por treinar com diferentes taxas de dropout.
			# "i" eh o valor a cada iteracao.
			for taxaDropout in valoresDropout:
			
				# Repetimos o treinamento algumas vezes para tirar uma media da eficiencia
				for iteracaoMedia in range(1,4):
					mnist = tf.keras.datasets.mnist
					(x_train, y_train),(x_test, y_test) = mnist.load_data()
					x_train, x_test = x_train / 255.0, x_test / 255.0
					model = tf.keras.models.Sequential([
					 tf.keras.layers.Flatten(),
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(10, activation=tf.nn.softmax)
					])
					model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
					 loss='sparse_categorical_crossentropy',
					 metrics=['accuracy'])
					model.fit(x_train, y_train, epochs=epocas)
					value = model.evaluate(x_test, y_test)
					model_json = model.to_json()
					json_file = open("model_MLP3.json", "w")
					json_file.write(model_json)
					json_file.close()
					model.save_weights("model_MLP3.h5")
					print("Model saved to disk")
					os.getcwd()
					
					somaDasEficienciasDeCadaIteracao = value[1] + somaDasEficienciasDeCadaIteracao
					
				
				
				myMutex.acquire()
				numeroDeNeuronios.append(neuronios)
				numeroDeEpocas.append(epocas)
				numeroDeCamadas.append(camadas)
				numeroDeDropout.append(taxaDropout)
				taxaDeAcertos.append(somaDasEficienciasDeCadaIteracao/iteracaoMedia)
				myMutex.release()
				
				# Reiniciamos a soma.
				somaDasEficienciasDeCadaIteracao = 0
				
# Vamos colocar uma thread para treinar cada rede com um numero especifico de camadas.
def thread4Camadas(camadas):
	
	# Para tirar a media das iteracoes, somaremos todas aqui e dividiremos pelo total.
	somaDasEficienciasDeCadaIteracao = 0
	
	# Os valores que utilizaremos para dropout variarao de 10% a 90% (instrucao abaixo).
	valoresDropout = range(10, 60, 10)# Variaremos de 10% em 10%.
	valoresDropout = [i/100 for i in valoresDropout]# Converte de porcentagem para escala de 0 a 1.
	
	# Testando resultados com diferentes quantidades de epocas.
	for epocas in [4, 8]:
	
		# Testando resultados com diferentes quantidades de neuronios.
		for neuronios in [256, 512]:
		
			# So para indicar em que passo da execucao estamos.
			print("\n\nepocas: " + str(epocas) + "\nCAMADAS" + str(camadas) + ": " + str(neuronios) + "\n\n")
	
			# Este loop fica responsável por treinar com diferentes taxas de dropout.
			# "i" eh o valor a cada iteracao.
			for taxaDropout in valoresDropout:
			
				# Repetimos o treinamento algumas vezes para tirar uma media da eficiencia
				for iteracaoMedia in range(1,4):
					mnist = tf.keras.datasets.mnist
					(x_train, y_train),(x_test, y_test) = mnist.load_data()
					x_train, x_test = x_train / 255.0, x_test / 255.0
					model = tf.keras.models.Sequential([
					 tf.keras.layers.Flatten(),
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(neuronios, activation=tf.nn.relu),
					 tf.keras.layers.Dropout(taxaDropout),# Diferentes valores de dropout.
					 tf.keras.layers.Dense(10, activation=tf.nn.softmax)
					])
					model.compile(optimizer=tf.train.GradientDescentOptimizer(0.17),
					 loss='sparse_categorical_crossentropy',
					 metrics=['accuracy'])
					model.fit(x_train, y_train, epochs=epocas)
					value = model.evaluate(x_test, y_test)
					model_json = model.to_json()
					json_file = open("model_MLP4.json", "w")
					json_file.write(model_json)
					json_file.close()
					model.save_weights("model_MLP4.h5")
					print("Model saved to disk")
					os.getcwd()
					
					somaDasEficienciasDeCadaIteracao = value[1] + somaDasEficienciasDeCadaIteracao
					
				
				
				myMutex.acquire()
				numeroDeNeuronios.append(neuronios)
				numeroDeEpocas.append(epocas)
				numeroDeCamadas.append(camadas)
				numeroDeDropout.append(taxaDropout)
				taxaDeAcertos.append(somaDasEficienciasDeCadaIteracao/iteracaoMedia)
				myMutex.release()
				
				# Reiniciamos a soma.
				somaDasEficienciasDeCadaIteracao = 0
				
				
	
if __name__ == '__main__':

	
	camadas1 = threading.Thread(target=thread1Camadas,args=(1,))
	camadas2 = threading.Thread(target=thread2Camadas,args=(2,))
	camadas3 = threading.Thread(target=thread3Camadas,args=(3,))
	camadas4 = threading.Thread(target=thread4Camadas,args=(4,))
	
	camadas1.start()
	camadas2.start()
	camadas3.start()
	camadas4.start()
	
	try:
		camadas4.join(); 
	except:
		pass;
		
	try:
		camadas3.join(); 
	except:
		pass;
		
	try:
		camadas2.join(); 
	except:
		pass;
		
	try:
		camadas1.join(); 
	except:
		pass;
		
	listasFile = open("listas.txt", "w")
	listasFile.write(str(numeroDeNeuronios) + "\n")
	listasFile.write(str(numeroDeEpocas) + "\n")
	listasFile.write(str(numeroDeCamadas) + "\n")
	listasFile.write(str(numeroDeDropout) + "\n")
	listasFile.write(str(taxaDeAcertos) + "\n")
	listasFile.close()
		
		
		
