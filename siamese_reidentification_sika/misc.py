import os
import zipfile

import random
import math
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Flatten, Dense, Dropout, BatchNormalization, Input, Lambda
from keras.metrics import Mean, CosineSimilarity
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet152, preprocess_input
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
#from keras.applications.vgg19 import VGG19, preprocess_input
#from keras.applications.efficientnet import EfficientNetB7, preprocess_input
import time
import pandas as pd
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score

def triplets(folder_paths, max_triplets= None):
    anchor_images = []
    positive_images = []
    negative_images = []

    for sika_folder in folder_paths:
        images = [os.path.join(sika_folder, img)
                  for img in os.listdir(sika_folder)]
        num_images = len(images)

        if num_images < 2:
            continue

        random.shuffle(images)

        for _ in range(max(num_images-1, max_triplets)):
            anchor_image = random.choice(images)

            positive_image = random.choice([x for x in images
                                            if x != anchor_image])

            negative_folder = random.choice([x for x in folder_paths
                                             if x != sika_folder])

            negative_image = random.choice([os.path.join(negative_folder, img)
                                            for img in os.listdir(negative_folder)])

            anchor_images.append(anchor_image)
            positive_images.append(positive_image)
            negative_images.append(negative_image)

    return anchor_images, positive_images, negative_images

def visualize_triplets(anchor_path, positive_path, negative_path, target_size = None):
    fig, axes = plt.subplots(1, 3, figsize = (10, 5))
    anchor_img = cv2.imread(anchor_path)
    positive_img = cv2.imread(positive_path)
    negative_img = cv2.imread(negative_path)

    # Resize images to the target size
    anchor_img = cv2.resize(anchor_img, target_size)
    positive_img = cv2.resize(positive_img, target_size)
    negative_img = cv2.resize(negative_img, target_size)

    axes[0].imshow(cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Anchor Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Positive Image")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Negative Image")
    axes[2].axis("off")            

def split_triplets(anchors,
                   positives,
                   negatives,
                   validation_split=0.2):

    triplets = list(zip(anchors, positives, negatives))

    train_triplets, val_triplets = train_test_split(triplets,
                                                    test_size=validation_split,
                                                    random_state=42)

    return train_triplets, val_triplets    

def visualize_triplet_samples(train_triplets, val_triplets, num_samples=None, target_size=None):
    num_cols = 6  # Adjust this to the number of images you want to display (2 for train and 2 for val)

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(2 * num_cols, 2 * num_samples))

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(num_samples):
        # Randomly select samples from the training set
        train_sample_index = random.randint(0, len(train_triplets) - 1)
        train_anchor, train_positive, train_negative = train_triplets[train_sample_index]

        train_anchor_img = cv2.imread(train_anchor)
        train_positive_img = cv2.imread(train_positive)
        train_negative_img = cv2.imread(train_negative)

        # Resize images to the target size
        train_anchor_img = cv2.resize(train_anchor_img, target_size)
        train_positive_img = cv2.resize(train_positive_img, target_size)
        train_negative_img = cv2.resize(train_negative_img, target_size)

        # Randomly select samples from the validation set
        val_sample_index = random.randint(0, len(val_triplets) - 1)
        val_anchor, val_positive, val_negative = val_triplets[val_sample_index]

        val_anchor_img = cv2.imread(val_anchor)
        val_positive_img = cv2.imread(val_positive)
        val_negative_img = cv2.imread(val_negative)

        # Resize images to the target size
        val_anchor_img = cv2.resize(val_anchor_img, target_size)
        val_positive_img = cv2.resize(val_positive_img, target_size)
        val_negative_img = cv2.resize(val_negative_img, target_size)

        # Plot images
        axes[i, 0].imshow(cv2.cvtColor(train_anchor_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Train Anchor")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.cvtColor(train_positive_img, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title("Train Positive")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cv2.cvtColor(train_negative_img, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title("Train Negative")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(cv2.cvtColor(val_anchor_img, cv2.COLOR_BGR2RGB))
        axes[i, 3].set_title("Val Anchor")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(cv2.cvtColor(val_positive_img, cv2.COLOR_BGR2RGB))
        axes[i, 4].set_title("Val Positive")
        axes[i, 4].axis("off")

        axes[i, 5].imshow(cv2.cvtColor(val_negative_img, cv2.COLOR_BGR2RGB))
        axes[i, 5].set_title("Val Negative")
        axes[i, 5].axis("off")

    plt.tight_layout()
    plt.show()

def load_and_preprocess_image(image_path, expand_dims=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    return image

def batch_generator(triplets, batch_size=None, augment=True):
    total_triplets = len(triplets)
    random_indices = list(range(total_triplets))
    random.shuffle(random_indices)
    
    datagen = ImageDataGenerator(
        rotation_range=10,  
        width_shift_range=0.05, 
        height_shift_range=0.05,   
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    for i in range(0, total_triplets, batch_size):
        batch_indices = random_indices[i:i + batch_size]
        batch_triplets = [triplets[j] for j in batch_indices]

        anchor_batch = []
        positive_batch = []
        negative_batch = []

        for triplet in batch_triplets:
            anchor, positive, negative = triplet
            
            anchor_image = load_and_preprocess_image(anchor)
            positive_image = load_and_preprocess_image(positive)
            negative_image = load_and_preprocess_image(negative)
                
            if augment:
                anchor_image = datagen.random_transform(anchor_image)
                positive_image = datagen.random_transform(positive_image)
                negative_image = datagen.random_transform(negative_image)

            anchor_batch.append(anchor_image)
            positive_batch.append(positive_image)
            negative_batch.append(negative_image)

        yield [np.array(anchor_batch),
               np.array(positive_batch),
               np.array(negative_batch)]

def get_embedding(input_shape, modeling = None, num_layers_to_unfreeze=25):
    base_model = modeling(weights='imagenet',
                                input_shape=input_shape,
                                include_top=False,
                                pooling='avg')

    for i in range(len(base_model.layers)-num_layers_to_unfreeze):
        base_model.layers[i].trainable = False

    embedding = tf.keras.models.Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128)
    ], name='Embedding')

    return embedding

class DistanceLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance
    
class SiameseModel(Model):
    def __init__(self, siamese_net, margin= 0.9):
        super().__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.loss_tracker = Mean(name='loss')
        self.accuracy_tracker = Mean(name='accuracy')

    def call(self, inputs):
        return self.siamese_net(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_net.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_net.trainable_weights)
        )

        self.loss_tracker.update_state(loss)

        accuracy = self._compute_accuracy(data)
        self.accuracy_tracker.update_state(accuracy)

        return {'loss': self.loss_tracker.result(),
                'accuracy': self.accuracy_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)

        accuracy = self._compute_accuracy(data)
        self.accuracy_tracker.update_state(accuracy)

        return {'loss': self.loss_tracker.result(),
                'accuracy': self.accuracy_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_net(data)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, .0)
        return loss

    def _compute_accuracy(self, data):
        ap_distance, an_distance = self.siamese_net(data)
        accuracy = tf.reduce_mean(tf.cast(ap_distance < an_distance,
                                          tf.float32))
        return accuracy

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def get_config(self):
        base_config = super().get_config()
        config = {
            'siamese_net': tf.keras.saving.serialize_keras_object(self.siamese_net),
            'margin': tf.keras.saving.serialize_keras_object(self.margin),
            'loss_tracker': tf.keras.saving.serialize_keras_object(self.loss_tracker),
            'accuracy_tracker': tf.keras.saving.serialize_keras_object(self.accuracy_tracker),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config['siamese_net'] = tf.keras.saving.deserialize_keras_object(config.pop('siamese_net'))
        config['margin'] = tf.keras.saving.deserialize_keras_object(config.pop('margin'))
        config['loss_tracker'] = tf.keras.saving.deserialize_keras_object(config.pop('loss_tracker'))
        config['accuracy_tracker'] = tf.keras.saving.deserialize_keras_object(config.pop('accuracy_tracker'))
        return cls(**config)    
    
def train_model(model,
                train_triplets,
                epochs,
                batch_size,
                val_triplets):

    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }

    train_steps_per_epoch = math.ceil(len(train_triplets) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_triplets) / batch_size)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train_loss = 0.
        train_accuracy = 0.
        val_loss = 0.
        val_accuracy = 0.

        with tqdm(total=train_steps_per_epoch, desc='Training') as pbar:
            for batch in batch_generator(train_triplets, batch_size=batch_size):
                loss, accuracy = model.train_on_batch(batch)
                train_loss += loss
                train_accuracy += accuracy

                pbar.update()
                pbar.set_postfix({'Loss': loss, 'Accuracy': accuracy})

        with tqdm(total=val_steps_per_epoch, desc='Validation') as pbar:
            for batch in batch_generator(val_triplets, batch_size=batch_size):
                loss, accuracy = model.test_on_batch(batch)
                val_loss += loss
                val_accuracy += accuracy

                pbar.update()
                pbar.set_postfix({'Loss': loss, 'Accuracy': accuracy})

        train_loss /= train_steps_per_epoch
        train_accuracy /= train_steps_per_epoch
        val_loss /= val_steps_per_epoch
        val_accuracy /= val_steps_per_epoch

        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')

    return model, history    

def save_metrics_to_csv(data, filename):
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print("Metrics saved successfully!")

def plot_model_history(history, fontsize= None, linewidth= None):
    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(history['loss'], linewidth=linewidth)
    plt.plot(history['val_loss'], linewidth=linewidth)
    plt.title('Model loss', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(['TRAIN', 'VAL'], loc='lower right', fontsize=fontsize)

    plt.subplot(122)
    plt.plot(history['accuracy'], linewidth=linewidth)
    plt.plot(history['val_accuracy'], linewidth=linewidth)
    plt.title('Model accuracy', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(['TRAIN', 'VAL'], loc='lower right', fontsize=fontsize)

    plt.show()    

def visualize_triplet_after_train(batch_generator, embedding):
    # Get the next batch
    anchor, positive, negative = next(batch_generator)

    # Select the first image from each batch
    anchor = anchor[0]
    positive = positive[0]
    negative = negative[0]

    # Expand dimensions to create a batch of size 1
    anchor = np.expand_dims(anchor, axis=0)
    positive = np.expand_dims(positive, axis=0)
    negative = np.expand_dims(negative, axis=0)

    # Obtain embeddings for anchor, positive, and negative samples
    anchor_embedding = embedding(preprocess_input(anchor))
    positive_embedding = embedding(preprocess_input(positive))
    negative_embedding = embedding(preprocess_input(negative))

    # Calculate cosine similarity
    cosine_similarity = CosineSimilarity()

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print(f'Positive similarity: {positive_similarity}')

    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    print(f'Negative similarity: {negative_similarity}')

    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(anchor[0])
    plt.title('Anchor')

    plt.subplot(1, 3, 2)
    plt.imshow(positive[0])
    plt.title('Positive')

    plt.subplot(1, 3, 3)
    plt.imshow(negative[0])
    plt.title('Negative')

    plt.show()

def save_anchor_positive_images(train_triplets, val_triplets):
    """
    Saves anchor and positive images with respective class name folders for training and validation sets.

    Parameters:
    - train_triplets: A list of triplets (anchor, positive, negative) for training.
    - val_triplets: A list of triplets (anchor, positive, negative) for validation.
    
    Returns:
    - train_copied_paths: A list of paths to the copied training anchor and positive images.
    - val_copied_paths: A list of paths to the copied validation anchor and positive images.
    """
    train_copied_paths = []
    val_copied_paths = []

    # Function to copy anchor and positive images to the respective folders
    def copy_images(triplets, directory):
        copied_paths = []
        for anchor_path, positive_path, _ in triplets:
            class_name = os.path.basename(os.path.dirname(anchor_path))  # Extract class name

            # Create class name folder if it doesn't exist in the specified directory
            class_folder = os.path.join(directory, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Get the filename of the anchor and positive images
            anchor_filename = os.path.basename(anchor_path)
            positive_filename = os.path.basename(positive_path)

            # Copy anchor and positive images to the respective class name folder
            copied_anchor_path = os.path.join(class_folder, anchor_filename)
            copied_positive_path = os.path.join(class_folder, positive_filename)

            shutil.copy(anchor_path, copied_anchor_path)
            shutil.copy(positive_path, copied_positive_path)

            copied_paths.append((copied_anchor_path, copied_positive_path))

        return copied_paths

    # Save anchor and positive images for training set
    train_copied_paths = copy_images(train_triplets, "train")

    # Save anchor and positive images for validation set
    val_copied_paths = copy_images(val_triplets, "valid")

    return train_copied_paths, val_copied_paths

# Define a function to extract embeddings for a given set of images
def extract_embeddings(image_paths, embedding_layer):
    embeddings = []
    for image_path in image_paths:
        # Load and preprocess the image
        image = load_and_preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Obtain the embedding for the image
        embedding = embedding_layer.predict(image)
        embeddings.append(embedding)
    return np.array(embeddings)

def visualize_embeddings(class_paths, embedding_layer):
    # Extract embeddings for each class
    class_embeddings = {}
    for class_path in class_paths:
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        class_name = os.path.basename(class_path)
        class_embeddings[class_name] = extract_embeddings(image_paths, embedding_layer)  # Provide the embedding_layer argument

    # Concatenate all embeddings and corresponding class labels
    all_embeddings = np.concatenate([embeddings for embeddings in class_embeddings.values()], axis=0)
    class_labels = np.concatenate([[class_name] * len(embeddings) for class_name, embeddings in class_embeddings.items()], axis=0)

    # Flatten the all_embeddings array
    flattened_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # Apply t-SNE to reduce the dimensionality of the embeddings
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(flattened_embeddings)

    # Plot t-SNE embeddings with class names as legends
    plt.figure(figsize=(10, 8))
    for class_name in class_embeddings.keys():
        indices = np.where(class_labels == class_name)
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=class_name)
    plt.title('t-SNE Visualization of Embeddings', fontsize=14)  # Adjust title font size
    plt.xlabel('t-SNE Dimension 1', fontsize=14)  # Adjust x-axis label font size
    plt.ylabel('t-SNE Dimension 2', fontsize=14)  # Adjust y-axis label font size
    plt.xticks(fontsize=14)  # Adjust x-axis tick font size
    plt.yticks(fontsize=14)  # Adjust y-axis tick font size
    plt.legend(fontsize=14)  # Adjust legend font size
    plt.show()

def save_embeddings_to_csv(class_paths, embedding_layer, output_filename):
    # Initialize lists to store data
    embeddings_list = []
    class_names_list = []

    # Extract embeddings for each class
    for class_path in class_paths:
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        class_name = os.path.basename(class_path)

        # Extract embeddings for images in the class
        for image_path in image_paths:
            # Load and preprocess the image
            image = load_and_preprocess_image(image_path)
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Obtain the embedding for the image
            embedding = embedding_layer.predict(image)
            
            # Append the embedding and class name to the lists
            embeddings_list.append(embedding.flatten())  # Flatten the embedding
            class_names_list.append(class_name)

    # Convert lists to numpy arrays
    embeddings_array = np.array(embeddings_list)
    class_names_array = np.array(class_names_list)

    # Create a DataFrame to store the data
    df = pd.DataFrame(embeddings_array)
    df['Class'] = class_names_array

    # Save the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)
    print(f"Embeddings saved to {output_filename} successfully!")