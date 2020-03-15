#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[ ]:


imgsrc=input("Enter The Link or Name Of The Image [if in same directory]")
im=cv2.imread(imgsrc)


# In[4]:


im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(im)


# In[5]:


#print(im.shape)
original_shape=im.shape
# that means the image is in 3D that is RGB values


# In[6]:


# Creating a 1 d array for every chanel i.e RGB
all_pixels=im.reshape(-1,3)


# In[7]:


#print(all_pixels.shape)
modified_shape=all_pixels.shape


# In[8]:


from sklearn.cluster import KMeans


# In[10]:


dominant_colors=int(input("Enter The Number Of Dominant Colors"))
km=KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)


# In[12]:


centers=km.cluster_centers_


# In[13]:


#print(centers)
# These are the dominant colours pixel values we have obtained from kmeans


# ###  As pixel values vary from 0-255 i.e. positive whole numbers of 8 bits 

# In[14]:


centers=np.array(centers,dtype='uint8')


# In[15]:


# Now these look as pixel values
#print(centers)


# In[ ]:





# In[16]:


print("Colors Which Are Dominant")
i=1
colors=[]
for each_col in centers:
    plt.subplot(1,4,i)
    plt.axis('Off')# TO remove the axis 
    i+=1
    colors.append(each_col)
    a=np.zeros((100,100,3),dtype='uint8')
    a[:,:,:]=each_col
    plt.imshow(a)
plt.show()


# ## Segmenting Our Original Image into The 4 Colours Extracted using Kmeans

# In[17]:


new_img=np.zeros(modified_shape,dtype='uint8')


# In[18]:


km.labels_


# In[19]:


for ix in range(new_img.shape[0]):
    new_img[ix]=colors[km.labels_[ix]]


# In[20]:


new_img=new_img.reshape(189,266,3)


# In[22]:


plt.imshow(new_img)
plt.show()


# In[ ]:




