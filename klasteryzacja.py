import os
import platform
import json
import numpy
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import OPTICS
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
# w listach x_video każdy element jest listą zawierającą podlistę w formacie [nazwa filmu, liczba klatek]
train_videos = []
test_videos = []
val_videos = []
#wczytywanie danych
for directory in os.scandir(data_dir+train_dir):
	n = 0
	for json_filename in os.scandir(directory):
		n = n + 1
		json_file = open(json_filename.path, "r")
		json_file_content = json_file.read()
		json_file.close()
		json_content = json.loads(json.dumps(json.loads(json_file_content)["people"]))[0]
		vector = json_content["pose_keypoints_2d"] + json_content["face_keypoints_2d"] + json_content["hand_left_keypoints_2d"] + json_content["hand_right_keypoints_2d"]
		train_array.append(vector)
	train_videos.append([directory,n])
for directory in os.scandir(data_dir+test_dir):
	n = 0
	for json_filename in os.scandir(directory):
		n = n + 1
		json_file = open(json_filename.path, "r")
		json_file_content = json_file.read()
		json_file.close()
		json_content = json.loads(json.dumps(json.loads(json_file_content)["people"]))[0]
		vector = json_content["pose_keypoints_2d"] + json_content["face_keypoints_2d"] + json_content["hand_left_keypoints_2d"] + json_content["hand_right_keypoints_2d"]
		test_array.append(vector)
	test_videos.append([directory,n])
for directory in os.scandir(data_dir+val_dir):
	n = 0
	for json_filename in os.scandir(directory):
		n = n + 1
		json_file = open(json_filename.path, "r")
		json_file_content = json_file.read()
		json_file.close()
		json_content = json.loads(json.dumps(json.loads(json_file_content)["people"]))[0]
		vector = json_content["pose_keypoints_2d"] + json_content["face_keypoints_2d"] + json_content["hand_left_keypoints_2d"] + json_content["hand_right_keypoints_2d"]
		val_array.append(vector)
	val_videos.append([directory,n])
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

# Utworzenie modelu z połączenia enkodera i dekodera (autoenkoder)
autoencoder = Model(inputs = input_dim, outputs = decoded13)

# Kompilacja modelu
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
# Uczenie modelu autoenkodera
autoencoder.fit(train_scaled, train_scaled, epochs = 3, batch_size = 32, shuffle = False, validation_data = (val_scaled,val_scaled))

# Wyodrębnienie modelu samego enkodera
encoder = Model(inputs = input_dim, outputs = encoded13)

# Enkodowanie zbioru train 
encoded_train = numpy.array(encoder.predict(train_scaled))

# Utworzenie modelu klasteryzacji (algorytm OPTICS) i dopasowanie do niego zbioru train
clustering = OPTICS(min_samples = 5).fit(encoded_train)
# Wyodrębnienie wektora z etykietami klastrów
labels = clustering.labels_
labels_set = set(labels)
# Utworzenie tablicy videos_converted, gdzie każdy element jest podlistą
# w postaci [nazwa filmu, lista etykiet kolejnych klatek]
train_videos_converted = []
counter = 0
for i in train_videos:
	frames_labelled = []
	for j in range(i[1]):
		frames_labelled.append(labels[counter+j])
	counter = counter + i[1]
	train_videos_converted.append([i[0],frames_labelled])

print("Number of unique labels:")
print(len(labels_set))
minus_ones = 0
for i in labels:
	if i==-1:
		minus_ones+=1
print("Number of -1 labels:")
print(minus_ones)


		
