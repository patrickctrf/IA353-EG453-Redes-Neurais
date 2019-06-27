import keras

mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(28, 28, 1),name='nome_exclusivo_00'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',name='nome_exclusivo_01'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),name='nome_exclusivo_02'))
model.add(keras.layers.Dropout(0.25,name='nome_exclusivo_03'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu',name='nome_exclusivo_04'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax',name='nome_exclusivo_05'))

model.get_config()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
evaluation = model.evaluate(x_test, y_test)

model.save('mnist_model.h5')
