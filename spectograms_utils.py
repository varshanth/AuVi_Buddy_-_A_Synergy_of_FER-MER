import os
import re
import PIL
from PIL import Image
import numpy as np
import pandas as pd
	
arousal_valence_file_tree = ["Music", "DEAM_Annotations", "annotations", "annotations averaged per song", "dynamic (per second annotations)"]
arousal_file = "arousal.csv"
valence_file = "valence.csv"

image_extension = "png"
_image_width = 300
_image_height = 300

'''
    Input 1: Full Path of Spectograms
	Input 2: Partial or Full Path of arousal and valence files
	Input 3: Max Images Per Sequence (Default: All)
    Input 4: Image Resolution in (Width, Height) (Default: Original)
	Input 5: number of entries (Default: 1744 which is the number of spectograms that are exactly 45 seconds long)
	
	Returns two values 
 		A 4 dimensional np array representing all spectograms 
 		A 3 dimensional np array representing each music/spectogram arousal valence values for 30 seconds
'''

def spectograms_get_data_set(_specto_path, 
							_arousal_valence_path,
							_max_im_per_seq = float('inf'),
							_image_resolution = (_image_width, _image_height),
							nEntries = 1744):
	
	# Validate Arguments
	contents_specto = set(os.listdir(_specto_path))
	
	# Check if directory actually contains spectogram images 
	_file_extension = re.compile('.*\.(.*)')
	_file_name_specto = re.compile('([a-z]+)[0-9]{1,4}.+\.' + image_extension)
	
	print("Checking spectogram directory...")
	for file in contents_specto:
		
		if(re.findall(_file_extension, file)[0] != image_extension):
			raise Exception('Invalid File Extension found')

		if(re.findall(_file_name_specto, file)[0] != "specto"):
			raise Exception('Invalid File Name found')
			
	print("Correct directory for spectograms")
	
	print("Checking arousal and valence directory...")
	# Get correct path for arousal and valence csv files
	contents_a_v = set(os.listdir(_arousal_valence_path))
	
	# print(contents_a_v)
	flagFinal = False
	
	intersect = set.intersection(contents_a_v, set(arousal_valence_file_tree))
	if(len(intersect) == 0):
		intersect = set.intersection(contents_a_v, {valence_file, arousal_file})
		if(len(intersect) == 2):
			flagFinal = True
		else:
			raise Exception('Invalid Directory for Valence and Arousal files')
			
	if(not flagFinal):
		intersect = list(intersect)[0]
		
		idx = arousal_valence_file_tree.index(intersect)
		
		# Build path to get the correct valence and arousal files
		for i in range(idx, len(arousal_valence_file_tree)):
			_arousal_valence_path += "\\" + arousal_valence_file_tree[i]
		
	# print(_arousal_valence_path)
	# Sanity check to see if files are actually in the directory
	new_contents_a_v = set(os.listdir(_arousal_valence_path))
	
	if(arousal_file not in new_contents_a_v):
		raise Exception('Could not find file ' + arousal_file)
	if(valence_file not in new_contents_a_v):
		raise Exception('Could not find file ' + valence_file)
		
	print("Correct directory for arousal and valence csv files")
	
	print("Processing valence and arousal data...")
	# Process arousal and valence files into np arrays
	# Read as dataframes 
	arousal = pd.read_csv(_arousal_valence_path + "\\" + arousal_file, nrows = nEntries) # Y axis
	valence = pd.read_csv(_arousal_valence_path + "\\" + valence_file, nrows = nEntries) # X axis
	
	# Read both files and store them as coordinates ---> ((x_0, y_0),(x_1, y_1),(x_2, y_2),...) -> That would correspond to one file.
	# At the end we'll have a 3 dimensional array 
	# Convert to numpy arrays
	arousal = arousal.values
	valence = valence.values

	arousal_valence_map = []
	
	for row in range(arousal.shape[0]):
		temp = []
		
		for col in range(1,61):
			
			coordinate = (valence[row, col], arousal[row, col])
			temp.append(coordinate)
			
		arousal_valence_map.append(temp)
		
	arousal_valence_map = np.array(arousal_valence_map)
	
	print("arousal_valence_map.shape:", arousal_valence_map.shape)
	print("Arousal Valence processing complete.")
	
	images  = []
	print("Processing spectograms...")
	
	# Convert images/spectograms to numpy arrays
	for file in contents_specto:
		
		imageFile = _specto_path + "\\" + file
		try:
		
			img = Image.open(imageFile)
			arr = np.array(img)
			images.append(arr)
			
		except:
			
			raise Exception('Error while trying to process ' + file)
			
	if(len(images) > 0):
		numImages = len(images)
		images = np.array(images)
		
		print("images.shape:", images.shape)
		print("Spectograms processing complete.")
		
		return images, arousal_valence_map
		
	else:
		raise Exception('Error while processing spectograms')
		
	
if __name__ == '__main__':
	
	spectograms_path = input("*** Enter path containing spectograms as png files ***\n")
	arousal_valence_path = input("*** Enter path containing arousal valence csv files ***\n" + 
								"*** It can be a partial file as long as it contains one of the following directories: ***\n" +
								"*** 'Music', 'DEAM_Annotations', 'annotations', 'annotations averaged per song', 'dynamic (per second annotations)' ***\n")
	
	X, Y = spectograms_get_data_set(spectograms_path, arousal_valence_path)
	print("**************************************************")
	print('Data Shape: {0}, Labels Shape: {1}'.format(X.shape, Y.shape))
	
	
	
	
	
	
	
	
	
	
	