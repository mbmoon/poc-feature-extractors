# Import modules
from cv2 import (imshow, waitKey, HOGDescriptor, SIFT_create, BOWKMeansTrainer,
                 BOWImgDescriptorExtractor, BFMatcher, NORM_L2, cvtColor, COLOR_RGB2GRAY, imread, IMREAD_GRAYSCALE)
from numpy import uint8, array, reshape, hsplit, vsplit
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn import svm
from sklearn.model_selection import train_test_split


def split_images(img_name, img_size):
    # Load the full image from the specified file
    img = imread(img_name, IMREAD_GRAYSCALE)

    # Find the number of sub-images on each row and column according to their size
    num_rows = img.shape[0] / img_size
    num_cols = img.shape[1] / img_size

    # Split the full image horizontally and vertically into sub-images
    sub_imgs = [hsplit(row, num_cols) for row in vsplit(img, num_rows)]

    return img, array(sub_imgs)


def build_img_array_from_path(path):
    files = Path(path).glob('*')
    # array_imgs = array([])
    array_imgs = []
    for fn in sorted(files):
        img = imread(str(fn), IMREAD_GRAYSCALE)
        img_prep = np.array(img)
        # array_imgs = np.append(array_imgs, img)
        array_imgs.append(img_prep)

    return array_imgs


def hog_descriptors(imgs, size, winSizeVar, blockSizeVar, cellSizeVar):
    # Create a list to store the HOG feature vectors
    hog_features = []

    # Set parameter values for the HOG descriptor based on the image data in use
    winSize = (winSizeVar, winSizeVar)
    blockSize = (blockSizeVar, blockSizeVar)
    blockStride = (int(blockSizeVar / 2), int(blockSizeVar / 2))
    cellSize = (cellSizeVar, cellSizeVar)
    nbins = 9

    # Set the remaining parameters to their default values
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64

    # Create a HOG descriptor
    hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # Compute HOG descriptors for the input images and append the feature vectors to the list
    for img in imgs:
        hist = hog.compute(img.reshape(size, size).astype(uint8))
        hog_features.append(hist)

    return array(hog_features)


def hog_visualization(img, descriptor, winSizeVar, blockSizeVar, cellSizeVar):
    winSize = (winSizeVar, winSizeVar)
    blockSize = (blockSizeVar, blockSizeVar)
    blockStride = (int(blockSizeVar / 2), int(blockSizeVar / 2))
    cell_size = (cellSizeVar, cellSizeVar)
    num_bins = 9

    print(blockStride)

    # Reshape the feature vector to [number of blocks in x and y direction, number of cells per block in x and y direction, number of bins]
    # This will be useful later when we plot the feature vector, so that the feature vector indexing matches the image indexing
    n_blockx = (winSize[0] // blockStride[0]) - 1
    n_blocky = (winSize[1] // blockStride[1]) - 1
    n_cellx = blockSize[0] // cell_size[0]
    n_celly = blockSize[1] // cell_size[1]

    hog_descriptor_reshaped = descriptor.reshape(n_blockx,
                                                 n_blocky,
                                                 n_cellx,
                                                 n_celly,
                                                 num_bins).transpose((1, 0, 2, 3, 4))

    # Create an array that will hold the average gradients for each cell
    ave_grad = np.zeros((n_blockx + 1, n_blocky + 1, num_bins))

    # Create an array that will count the number of histograms per cell
    hist_counter = np.zeros((n_blockx + 1, n_blocky + 1, 1))

    # Add up all the histograms for each cell and count the number of histograms per cell
    for i in range(n_cellx):
        for j in range(n_celly):
            ave_grad[i:n_blockx + i,
            j:n_blocky + j] += hog_descriptor_reshaped[:, :, i, j, :]

            hist_counter[i:n_blockx + i,
            j:n_blocky + j] += 1

    # Calculate the average gradient for each cell
    # print(hist_counter)
    ave_grad /= hist_counter

    # Calculate the total number of vectors we have in all the cells.
    len_vecs = ave_grad.shape[0] * ave_grad.shape[1] * ave_grad.shape[2]

    # Create an array that has num_bins equally spaced between 0 and 180 degress in radians.
    deg = np.linspace(0, np.pi, num_bins, endpoint=False)

    # Each cell will have a histogram with num_bins. For each cell, plot each bin as a vector (with its magnitude
    # equal to the height of the bin in the histogram, and its angle corresponding to the bin in the histogram).
    # To do this, create rank 1 arrays that will hold the (x,y)-coordinate of all the vectors in all the cells in the
    # image. Also, create the rank 1 arrays that will hold all the (U,V)-components of all the vectors in all the
    # cells in the image. Create the arrays that will hold all the vector positons and components.
    U = np.zeros((len_vecs))
    V = np.zeros((len_vecs))
    X = np.zeros((len_vecs))
    Y = np.zeros((len_vecs))

    # Set the counter to zero
    counter = 0

    # Use the cosine and sine functions to calculate the vector components (U,V) from their maginitudes. Remember the
    # cosine and sine functions take angles in radians. Calculate the vector positions and magnitudes from the
    # average gradient array
    for i in range(ave_grad.shape[0]):
        for j in range(ave_grad.shape[1]):
            for k in range(ave_grad.shape[2]):
                U[counter] = ave_grad[i, j, k] * np.cos(deg[k])
                V[counter] = ave_grad[i, j, k] * np.sin(deg[k])

                X[counter] = (cell_size[0] / 2) + (cell_size[0] * i)
                Y[counter] = (cell_size[1] / 2) + (cell_size[1] * j)

                counter = counter + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    ax1.set(title='Grayscale Image')
    ax1.imshow(img, cmap='gray')

    # Plot the feature vector (HOG Descriptor)
    ax2.set(title='HOG Descriptor')
    ax2.quiver(Y, X, U, V, color='white', headwidth=0, headlength=0, scale_units='inches', scale=3)
    ax2.invert_yaxis()
    ax2.set_aspect(aspect=1)
    ax2.set_facecolor('black')

    plt.show()

sub_imgs = build_img_array_from_path("Images/RSSCN7/fResident")
sub_imgs_labels = [True] * len(sub_imgs)
sub_imgs_dataset = list(zip(sub_imgs, sub_imgs_labels))

sub_imgs_false = build_img_array_from_path("Images/RSSCN7/gParking")
sub_imgs_false_labels = [False] * len(sub_imgs)
sub_imgs_false_dataset = list(zip(sub_imgs_false, sub_imgs_false_labels))

sub_imgs_conc = sub_imgs_dataset + sub_imgs_false_dataset
random.shuffle(sub_imgs_conc)

# for i in sub_imgs_conc[:5]:
#     print(i)
#print(sub_imgs_false_dataset[4])

# print(sub_imgs.shape[0])
# print(sub_imgs.shape[1])

# cv2.imshow("Resized image", sub_imgs[400])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)

# rs_image_hog = hog_descriptors(sub_imgs, 400)
# print('Size of HOG feature vectors:', rs_image_hog.shape)

# print(rs_image_hog[33])

# plt.imshow(rs_image_hog[1][51])

# img = sub_imgs[33]

# configuration
size = 400
winSizeVar = 400
blockSizeVar = 100
cellSizeVar = 100

# img = imread('Images/zero.jpg', IMREAD_GRAYSCALE)
# rs_image_hog = hog_descriptors([img], size, winSizeVar, blockSizeVar, cellSizeVar)
# print('Size of HOG feature vectors:', rs_image_hog.shape)

# img = sub_imgs[33]
# rs_image_hog = hog_descriptors([img], size, winSizeVar, blockSizeVar, cellSizeVar)
# print('Size of HOG feature vectors:', rs_image_hog.shape)

#рog_visualization(img, rs_image_hog, winSizeVar, blockSizeVar, cellSizeVar)

dataset = sub_imgs_conc

validation_dataset = build_img_array_from_path("Images/RSSCN7_test")

validation_data = []

for validation in validation_dataset:
    descriptor = hog_descriptors([validation], size, winSizeVar, blockSizeVar, cellSizeVar)
    #print('Size of HOG feature vectors:', descriptor.shape)
    validation_data.append(descriptor[0])

# Compute the HOG descriptors for the images
#shog = cv2.HOGDescriptor()
data = []
labels = []

for image, label in dataset:
    descriptor = hog_descriptors([image], size, winSizeVar, blockSizeVar, cellSizeVar)
    #print('Size of HOG feature vectors:', descriptor.shape)
    data.append(descriptor[0])
    labels.append(label)

# Convert the data to a numpy array
data = np.array(data)
labels = np.array(labels)

# Split the dataset into a training set and a test set
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print('Shape of datatrain:', data_train.shape)
print('Shape of labels_train:',labels_train.shape)

# Train the SVM
clf = svm.SVC()
clf.fit(data_train, labels_train)

print(data_test.shape)
# Test the SVM
labels_pred = clf.predict(data_test)

labels_validation = clf.predict(validation_data)

# Print the accuracy
accuracy = np.sum(labels_pred == labels_test) / labels_test.shape[0]
print('Accuracy: ', accuracy)

print(labels_validation)
