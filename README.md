# Object_detection_using_unet

Sure! Here's your original segmentation-based object detection code broken into steps with concise explanations, following your specified format.

---

## Step 1: Import Libraries

```python
import numpy as np  
import tensorflow as tf  
from tensorflow.keras import layers, models  
import matplotlib.pyplot as plt  
from PIL import Image  
import os  
from tqdm import tqdm  
import albumentations as A  
```

## Explanation:

* Loads libraries for data handling, image processing, model creation, training, and augmentation.

---

## Step 2: Load Image and Mask Filenames

```python
images = os.listdir("images/")  
masks = os.listdir("masks/")
```

## Explanation:

* Reads filenames of input images and their corresponding masks.

---

## Step 3: Preprocessing Function

```python
def create_dataset(img_path, mask_path, img_size=192):  
    ...  
    return xtrain, ytrain
```

## Explanation:

* Resizes images and masks to uniform shape (192×192).
* Converts image pixels to 0–255 range and masks to single-channel.

---

## Step 4: Prepare Dataset

```python
x, y = create_dataset("images/", "masks/")  
y = y.astype(bool)
```

## Explanation:

* Loads and preprocesses image/mask pairs.
* Converts masks to binary (boolean) for segmentation.

---

## Step 5: Define U-Net Model

```python
inp = layers.Input((192, 192, 3))  
...  
model = models.Model(inputs=[inp], outputs=[output])  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Explanation:

* Implements U-Net architecture with skip connections.
* Uses binary crossentropy for segmentation of foreground vs. background.

---

## Step 6: Train the Model

```python
model.fit(x, y, epochs=10, batch_size=8, validation_split=0.2)
```

## Explanation:

* Trains the U-Net on image-mask pairs for 10 epochs with 20% validation.

---

## Step 7: Post-processing – Mask to Bounding Box

```python
def overlap(img, mask):  
    ...  
    plt.imshow(cropped)
```

## Explanation:

* Converts predicted mask to bounding box using non-zero pixel positions.
* Draws green overlay and shows cropped object region.

---

## Step 8: Predict and Visualize

```python
ypre = model.predict(x[200:210])  
overlap(x[202], ypre[2])
```

## Explanation:

* Predicts segmentation masks and visualizes bounding box from mask.

---

## Step 9: Define Albumentations Transform

```python
train_transform = A.Compose([...], bbox_params=A.BboxParams(...))
```

## Explanation:

* Applies image augmentations (e.g. flip, rain, contrast) while preserving bounding boxes in YOLO format.

---

## Step 10: Augment Dataset

```python
mfactor = 4  
...  
for i in tqdm(range(xtrain_2.shape[0])):  
    ...
```

## Explanation:

* Augments dataset by creating `mfactor` versions of each image with corresponding bounding boxes.
* Increases data diversity to improve model robustness.

---

## Step 11: Clean Up Dataset

```python
X_trans = X_trans[:-4]  
Y_trans = Y_trans[:-4]
```

## Explanation:

* Removes excess padded samples at the end (if any).

---

## Step 12: Visualize Augmented Output

```python
out = post_processing([X_trans[-1]], [Y_trans[-1]])
```

## Explanation:

* Displays augmented image with its updated bounding box.

---

Let me know if you’d like to modularize the code or convert it into a training pipeline script!
