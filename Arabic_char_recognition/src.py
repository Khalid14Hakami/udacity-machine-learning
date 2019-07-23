import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd 





# loading data
x_train = pd.read_csv('./data/csvTrainImages 13440x1024.csv', header = None)
y_train = pd.read_csv('./data/csvTrainLabel 13440x1.csv', header = None)
x_test = pd.read_csv('./data/csvTestImages 3360x1024.csv', header = None)
y_test = pd.read_csv('./data/csvTestLabel 3360x1.csv', header = None)

# reading the values
x_train = x_train.iloc[:,:].values.astype('float32')
y_train = y_train.iloc[:,:].values.astype('int32')-1
x_test = x_test.iloc[:,:].values.astype('float32')
y_test = y_test.iloc[:,:].values.astype('int32')-1

# Reshaping arrays 
x_train = x_train.reshape(x_train.shape[0], 32, 32)
x_train = x_train.swapaxes(1, 2)
x_test = x_test.reshape(x_test.shape[0], 32, 32)
x_test = x_test.swapaxes(1, 2)


# Normalizing 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Building the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(28, activation=tf.nn.softmax))

# Compile the model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=16)

# testing the accuracy 
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)



