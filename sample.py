import os

# Specify the path to the folder containing the images
folder_path = 'Basis Image'

# Get a list of all image filenames in the folder
image_filenames = os.listdir(folder_path)

# Specify the index of the image you want to retrieve
image_index = 2

# Get the filename of the image at the specified index
image_filename = image_filenames[image_index]

# Print the filename of the image at the specified index
print(image_filename)
