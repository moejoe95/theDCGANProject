import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import os
import time
import numpy as np

import generator as gen
import descriminator as des
import dataset as ds

from cv2 import cv2
from IPython import display


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = gen.generator_loss(fake_output)
        disc_loss = des.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        print('run epoch ' + str(epoch))
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    _ = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        sample = cv2.cvtColor(predictions[i, :, :, ].numpy() * .5 + .5, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 4, i + 1)
        plt.imshow(sample)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# Display a single image using the epoch number
def display_image(epoch_no):
    return cv2.imread('image_at_epoch_{:04d}.png'.format(epoch_no))


# get data
train_images = ds.get_dataset_folders(['lsunbed_dataset'], (128, 128), recursive=True)

# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
np.subtract(train_images, .5, out=train_images)
np.divide(train_images, .5, out=train_images)

for i in range(16):
    sample = cv2.cvtColor(train_images[i, :, :, ] * .5 + .5, cv2.COLOR_BGR2RGB)
    plt.subplot(4, 4, i + 1)
    plt.imshow(sample)
    plt.axis('off')
plt.show()

BUFFER_SIZE = 60000
BATCH_SIZE = 32

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = gen.make_generator_model_big()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = des.make_discriminator_model_big()
decision = discriminator(generated_image)

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

checkpoint_dir = './training_checkpoints_lsun_big'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)

anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

display.Image(filename=anim_file)
