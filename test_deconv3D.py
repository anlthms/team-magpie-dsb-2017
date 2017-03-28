from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.convolutional import Convolution3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from new.deconv3D import Deconvolution3D


import numpy as np
import pylab as plt

filename = 'model'

_shape = (16,16)
# _shape = (128,128)
# _shape = (64,64)

# time_batch_sz = (None,)
time_batch_sz = (15,)
batch_sz = (10,)

x = Input(batch_shape=(batch_sz + time_batch_sz +_shape + (1,)))

conv1 = Convolution3D(nb_filter=5, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                   border_mode='same', subsample=(1, 2, 2))

conv2 = Convolution3D(nb_filter=10, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                   border_mode='same', subsample=(1, 2, 2))

out_shape_2 = (10, 15, 8, 8, 10)
dconv1 = Deconvolution3D(nb_filter=10, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, output_shape=out_shape_2,
                   border_mode='same', subsample=(1, 1, 1))

out_shape_1 = (10, 16, 17, 17, 5)
dconv2 = Deconvolution3D(nb_filter=5, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, output_shape=out_shape_1,
                   border_mode='same', subsample=(1, 1, 1))

decoder_squash = Convolution3D(1, 2, 2, 2, border_mode='valid', activation='sigmoid')

out = decoder_squash(dconv2(dconv1(conv2(conv1(x)))))

seq = Model(x,out)
seq.compile(loss='mse', optimizer='adadelta')

seq.summary(line_length=150)


# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (_shape*2)
# and at the end we select a 40x40 window.

_shape = (16,16)

def generate_movies(n_samples=1200, n_frames=15):
    row = _shape[0]*2
    col = _shape[1]*2
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    x_clip_st = _shape[0]-_shape[0]/2
    x_clip_ed = _shape[0]+x_clip_st
    y_clip_st = _shape[0]-_shape[0]/2
    y_clip_ed = _shape[0]+y_clip_st
    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(x_clip_st, x_clip_ed)
            ystart = np.random.randint(y_clip_st, y_clip_ed)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, x_clip_st:x_clip_ed, y_clip_st:y_clip_ed, ::]
    shifted_movies = shifted_movies[::, ::, x_clip_st:x_clip_ed, y_clip_st:y_clip_ed, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

# Train the network
noisy_movies, shifted_movies = generate_movies(n_samples=1200)


checkpointer = []
checkpointer.append(EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'))


print noisy_movies.shape
print shifted_movies.shape

seq.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=10,
        nb_epoch=300, validation_split=0.05, callbacks=checkpointer)
seq.save_weights('{0}_final_wts.h5'.format(filename))


# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))

