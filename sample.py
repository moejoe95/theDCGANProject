import tensorflow as tf

import matplotlib.pyplot as plt
import os
import numpy as np

import generator as gen
import descriminator as des

from cv2 import cv2


generator = gen.make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = des.make_discriminator_model()
decision = discriminator(generated_image)

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

checkpoint_dir = './training_checkpoints_lsun'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

noise_dim = 100
seed = tf.random.normal([2, noise_dim])

seed = seed.numpy()

# interpolate in n steps
interpolated_seeds = []
steps = 9
for i in range(10):
    interpolated_seeds.append((steps - i) * (1/steps) * seed[0] + i * (1/steps) * seed[1])

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

interpolated_seeds = np.asarray(interpolated_seeds)
predictions = generator(interpolated_seeds, training=False)

for i in range(10):
    samplebgr = predictions[i, :, :, ].numpy() * .5 + .5
    sample = cv2.cvtColor(samplebgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite('sample_' + str(i) + '.png', samplebgr * 255)
    plt.imshow(sample)
    plt.axis('off')
    plt.show()
