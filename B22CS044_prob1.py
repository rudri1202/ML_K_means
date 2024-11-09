#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy matplotlib scikit-learn pillow')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
import random
from PIL import Image


# In[3]:


image_path = 'test.png'
image = Image.open(image_path)
image


# In[4]:


image_np = np.array(image)
image_reshaped = image_np.reshape(-1, 3)
print(image_reshaped.shape)
image_np.shape


# In[5]:


def init_centroids(num_clusters, data):
   
    m, _ = data.shape
    centroids_init = np.empty([num_clusters, 3])

    for i in range(num_clusters):
        rand_idx = random.randint(0, m - 1)
        centroids_init[i] = data[rand_idx]

    return centroids_init


# In[6]:


def computeCentroid(features):

  return np.mean(features, axis=0)


# In[7]:


def mykmeans(centroids, data, max_iter=5):
   
    m = data.shape[0]

    for i in range(max_iter):
        centroid_rgbs = {}

        for j in range(m):
            centroid = np.argmin(np.linalg.norm(centroids - data[j], axis=1))
            if centroid in centroid_rgbs:
                centroid_rgbs[centroid] = np.append(centroid_rgbs[centroid], [data[j]], axis=0)
            else:
                centroid_rgbs[centroid] = np.array([data[j]])

        for k in centroid_rgbs:
            centroids[k] = computeCentroid(centroid_rgbs[k])

    return centroids


# In[8]:


def update_image(data, centroids):
    # Get the number of data points
    m = data.shape[0]
    
    # Create an empty array to store the new data
    new_data = np.empty_like(data)

    # Iterate over each data point
    for i in range(m):
        # Find the index of the nearest centroid for the current data point
        nearest_centroid = np.argmin(np.linalg.norm(centroids - data[i], axis=1))
        # Update the current data point with the coordinates of the nearest centroid
        new_data[i] = centroids[nearest_centroid]

    return new_data


# In[9]:


def compress_image(image_data, centroids):
    labels = np.argmin(np.linalg.norm(centroids - image_data[:, np.newaxis], axis=2), axis=0)
    compressed_image = centroids[labels]
    compressed_image = compressed_image.reshape(image_data.shape)
    compressed_image = compressed_image.astype(np.uint8)
    return compressed_image


# In[10]:


def display_images(original_img, compressed_img, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(compressed_img)
    ax[1].set_title(title)
    ax[1].axis('off')
    plt.show()


# In[11]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Define the number of clusters
num_clusters_list = [2, 4, 8, 20]

# Load the image and reshape it
image_path = 'test.png'
image = mpimg.imread(image_path)
image_reshaped = image.reshape(-1, 3)

# Iterate over different numbers of clusters
for num_clusters in num_clusters_list:
    # Initialize centroids
    initial_centroids = init_centroids(num_clusters, image_reshaped)
    
    # Perform K-means clustering
    final_centroids = mykmeans(initial_centroids, image_reshaped, max_iter=5)
    
    # Update image data with the final centroids
    image_compressed = update_image(image_reshaped, final_centroids)

    # Display the compressed image
    plt.imshow(image_compressed.reshape(image.shape))
    plt.axis('off')
    plt.title(f'Compressed Image (k={num_clusters})')
    plt.savefig(fname=f'monkey_compressed_{num_clusters}_clusters.png', format='png', dpi=300)
    plt.show()


# In[12]:


image_path = 'test.png'
image = Image.open(image_path)
image_np = np.array(image)
image_reshaped = image_np.reshape(-1, 3)
print(image_reshaped.shape)
image_np.shape


# In[13]:


kmeans = KMeans(n_clusters=6, random_state=0).fit(image_reshaped)
labels = kmeans.predict(image_reshaped)
centroids = kmeans.cluster_centers_


# In[14]:


compressed_image = centroids[labels]
compressed_image = compressed_image.reshape(image_np.shape)
compressed_image = compressed_image.astype(np.uint8)
fig, ax = plt.subplots(1, 3, figsize=(21, 21))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(compressed_image)
ax[1].set_title('Compressed Image using lib({} colors)'.format(6))
ax[1].axis('off')

ax[2].imshow(image_compressed.reshape(image_np.shape))
ax[2].set_title('compressed image from scratch')
ax[2].axis('off')

plt.show()


# In[15]:


def spatial_distance(pixel1, pixel2):
    # Calculate Euclidean distance between pixel coordinates
    return np.sqrt(np.sum((pixel1 - pixel2) ** 2))

def mykmeans_with_spatial_coherence(centroids, data, max_iter=5, spatial_weight=0.5):
    m = data.shape[0]
    
    for i in range(max_iter):
        centroid_rgbs = {}
        
        for j in range(m):
            # Calculate distances to centroids
            distances = np.linalg.norm(centroids - data[j], axis=1)
            # Calculate spatial distances to centroids
            spatial_distances = [spatial_distance(centroids[k], data[j]) for k in range(len(centroids))]
            # Combine color and spatial distances
            total_distances = distances + spatial_weight * np.array(spatial_distances)
            # Assign pixel to nearest centroid
            centroid = np.argmin(total_distances)
            if centroid in centroid_rgbs:
                centroid_rgbs[centroid] = np.append(centroid_rgbs[centroid], [data[j]], axis=0)
            else:
                centroid_rgbs[centroid] = np.array([data[j]])
        
        for k in centroid_rgbs:
            centroids[k] = computeCentroid(centroid_rgbs[k])
    
    return centroids

# Usage
initial_centroids = init_centroids(num_clusters, image_reshaped)
final_centroids_with_spatial = mykmeans_with_spatial_coherence(initial_centroids, image_reshaped, max_iter=5, spatial_weight=0.5)
image_compressed_with_spatial = update_image(image_reshaped, final_centroids_with_spatial)


# In[16]:


plt.imshow(image_compressed_with_spatial.reshape(image_np.shape))
plt.title('Compressed Image with Spatial Coherence')
plt.axis('off')
plt.show()

