import os
import platform
import json
import numpy
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from keras.layers import Input, Dense
from keras.models import Model

platform_name = platform.system()
data_dir = "PB"
if platform_name == "Windows":
	train_dir = "\\train\\openpose_output\\json"
	test_dir = "\\test\\openpose_output\\json"
	val_dir = "\\val\\openpose_output\\json"
else:
	train_dir = "/train/openpose_output/json"
	test_dir = "/test/openpose_output/json"
	val_dir = "/val/openpose_output/json"
train_array = []
test_array = []
val_array = []
#wczytywanie danych
for directory in os.scandir(data_dir+train_dir):
	for json_filename in os.scandir(directory):
		json_file = open(json_filename.path, "r")
		json_file_content = json_file.read()
		json_file.close()
		json_content = json.loads(json.dumps(json.loads(json_file_content)["people"]))[0]
		vector = json_content["pose_keypoints_2d"] + json_content["face_keypoints_2d"] + json_content["hand_left_keypoints_2d"] + json_content["hand_right_keypoints_2d"]
		train_array.append(vector)
for directory in os.scandir(data_dir+test_dir):
	for json_filename in os.scandir(directory):
		json_file = open(json_filename.path, "r")
		json_file_content = json_file.read()
		json_file.close()
		json_content = json.loads(json.dumps(json.loads(json_file_content)["people"]))[0]
		vector = json_content["pose_keypoints_2d"] + json_content["face_keypoints_2d"] + json_content["hand_left_keypoints_2d"] + json_content["hand_right_keypoints_2d"]
		test_array.append(vector)
for directory in os.scandir(data_dir+val_dir):
	for json_filename in os.scandir(directory):
		json_file = open(json_filename.path, "r")
		json_file_content = json_file.read()
		json_file.close()
		json_content = json.loads(json.dumps(json.loads(json_file_content)["people"]))[0]
		vector = json_content["pose_keypoints_2d"] + json_content["face_keypoints_2d"] + json_content["hand_left_keypoints_2d"] + json_content["hand_right_keypoints_2d"]
		val_array.append(vector)
		
train_array = numpy.array(train_array)
test_array = numpy.array(test_array)
val_array = numpy.array(val_array)
#normalizacja
train_scaled = minmax_scale(train_array, axis = 0)
test_scaled = minmax_scale(test_array, axis = 0)
val_scaled = minmax_scale(val_array, axis = 0)

n_attributes = train_scaled.shape[1]

#docelowa liczba wymiarów dla enkodera
encoding_dim = 200
input_dim = Input(shape = (n_attributes, ))

# Warstwy enkodera
encoded1 = Dense(3000, activation = 'relu')(input_dim)
encoded2 = Dense(2750, activation = 'relu')(encoded1)
encoded3 = Dense(2500, activation = 'relu')(encoded2)
encoded4 = Dense(2250, activation = 'relu')(encoded3)
encoded5 = Dense(2000, activation = 'relu')(encoded4)
encoded6 = Dense(1750, activation = 'relu')(encoded5)
encoded7 = Dense(1500, activation = 'relu')(encoded6)
encoded8 = Dense(1250, activation = 'relu')(encoded7)
encoded9 = Dense(1000, activation = 'relu')(encoded8)
encoded10 = Dense(750, activation = 'relu')(encoded9)
encoded11 = Dense(500, activation = 'relu')(encoded10)
encoded12 = Dense(250, activation = 'relu')(encoded11)
encoded13 = Dense(encoding_dim, activation = 'relu')(encoded12)

# Warstwy enkodera
decoded1 = Dense(250, activation = 'relu')(encoded13)
decoded2 = Dense(500, activation = 'relu')(decoded1)
decoded3 = Dense(750, activation = 'relu')(decoded2)
decoded4 = Dense(1000, activation = 'relu')(decoded3)
decoded5 = Dense(1250, activation = 'relu')(decoded4)
decoded6 = Dense(1500, activation = 'relu')(decoded5)
decoded7 = Dense(1750, activation = 'relu')(decoded6)
decoded8 = Dense(2000, activation = 'relu')(decoded7)
decoded9 = Dense(2250, activation = 'relu')(decoded8)
decoded10 = Dense(2500, activation = 'relu')(decoded9)
decoded11 = Dense(2750, activation = 'relu')(decoded10)
decoded12 = Dense(3000, activation = 'relu')(decoded11)
decoded13 = Dense(n_attributes, activation = 'sigmoid')(decoded12)

# Utworzenie modelu z połączenia enkodera i enkodera (autoenkoder)
autoencoder = Model(inputs = input_dim, outputs = decoded13)

# Kompilacja modelu
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
# Uczenie modelu autoenkodera
autoencoder.fit(train_scaled, train_scaled, epochs = 10, batch_size = 32, shuffle = False, validation_data = (val_scaled,val_scaled))

# Wyodrębnienie modelu samego enkodera
encoder = Model(inputs = input_dim, outputs = encoded13)

# Otrzymanie wynikowych zbiorów testowych na wyjściu samego enkodera i autoenkodera
encoded_test = numpy.array(encoder.predict(test_scaled))
autoencoded_test = numpy.array(autoencoder.predict(test_scaled))

# Porównanie wymiarów
print("Dimensions:")
print("Original:")
print(test_scaled.shape)
print("Encoded:")
print(encoded_test.shape)
print("Decoded:")
print(autoencoded_test.shape)
print("-----------")

#Prosty test na to jak bardzo różnią się dane wejściowe od zdekodowanych przez autoenkoder
avg = 0
for i in range(test_scaled.shape[0]):
	for j in range(n_attributes):
		diff = abs(test_scaled[i][j] - autoencoded_test[i][j])
		avg+=diff
avg = avg / (n_attributes * test_scaled.shape[0]) 
print(avg)


		
