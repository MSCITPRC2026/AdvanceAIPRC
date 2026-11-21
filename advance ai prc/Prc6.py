import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the range [-1, 1] for better GAN performance
x_train = x_train.astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension (1, height, width, 1)

# Set parameters
latent_dim = 100  # Dimension of the latent space (noise)
batch_size = 64
epochs = 10000
save_interval = 1000

# Create the Generator model
def build_generator():
    model = tf.keras.Sequential()

    # Fully connected layer with ReLU activation
    model.add(layers.Dense(128 * 7 * 7, input_dim=latent_dim))  # Output enough features for reshape
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    # Reshape the output of the Dense layer into a 3D tensor (7x7x128)
    model.add(layers.Reshape((7, 7, 128)))  # 7x7 image with 128 filters

    # Upsample to 14x14
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    # Upsample to 28x28
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    # Final layer: 28x28 image with tanh activation
    model.add(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))

    return model

# Create the Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()

    # Convolutional layers with LeakyReLU activations
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))  # Output a single value (real/fake)

    return model

# Build the GAN model by combining the Generator and the Discriminator
def build_gan(generator, discriminator):
    # Freeze the discriminator during GAN training (we only train the generator)
    discriminator.trainable = False

    gan_input = layers.Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = tf.keras.Model(gan_input, gan_output)
    return gan

# Compile the Discriminator and GAN
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator()
gan = build_gan(generator, discriminator)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Function to train the GAN
def train_gan(epochs, batch_size, save_interval):
    half_batch = batch_size // 2  # Half the batch size for real and fake

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))  # Change to match full batch size
    fake = np.zeros((batch_size, 1))  # Change to match full batch size

    for epoch in range(epochs):

        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_images = x_train[idx]

        # Generate fake images
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_images = generator.predict(noise)

        # Train on real images
        d_loss_real = discriminator.train_on_batch(real_images, valid[:half_batch])  # Use only half batch for real
        # Train on fake images
        d_loss_fake = discriminator.train_on_batch(generated_images, fake[:half_batch])  # Use only half batch for fake
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator via GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)  # Use full batch for generator

        # Print the progress
        if epoch % save_interval == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            save_generated_images(epoch)

# Function to save generated images
def save_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale images to [0, 1]

    # Plot and save images
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"gan_generated_image_{epoch}.png")
    plt.close()

# Train the GAN
train_gan(epochs, batch_size, save_interval)
