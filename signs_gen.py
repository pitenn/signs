import numpy as np
from skimage import color, exposure, transform, io
from pickle import dump, load
import tensorflow as tf
from keras import backend as K
import glob
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

JOBS = 6

def configure_usage_of_threads(jobs):
    config = tf.ConfigProto(intra_op_parallelism_threads=jobs, inter_op_parallelism_threads=jobs,  allow_soft_placement=True, device_count = {'CPU': jobs})
    session = tf.Session(config=config)
    K.set_session(session)

NUM_CLASSES = 62
IMG_SIZE = 100


def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2, :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


def process_and_save_images(root_dir, x_axis_path, y_axis_path):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with open(x_axis_path, mode='wb') as f:
        dump(X, f)

    with open(y_axis_path, mode='wb') as f:
        dump(Y, f)

    return X, Y

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


if __name__ == "__main__":
    configure_usage_of_threads(JOBS)
    root_dir = 'Training/'
    imgs = []
    labels = []

    x_axis_filename = "X_AXIS.bin"
    y_axis_filename = "Y_AXIS.bin"

    X = None
    Y = None

    if os.path.exists(x_axis_filename) and os.path.exists(y_axis_filename):
        with open(x_axis_filename, mode="rb") as f:
            X = load(f)
        with open(y_axis_filename, mode="rb") as f:
            Y = load(f)
    else:
        X, Y = process_and_save_images(root_dir, x_axis_filename, y_axis_filename)

    K.set_image_data_format('channels_first')
    model = cnn_model()

    # let's train the model using SGD + momentum
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    def lr_schedule(epoch):
        return lr * (0.1 ** int(epoch / 10))

    batch_size = 32
    epochs = 9

    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint('model.h5', save_best_only=True)]
              )

    with open("model.json", mode="w") as f:
        f.write(model.to_json())
        print("Saved model JSON")
    model.save_weights("model.h5")

    model.predict(  )
    print("Saved model H5 weights.")
