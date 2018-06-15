from keras.models import load_model, model_from_json
import json
import os
import glob
from signs_gen import *
from pickle import dump, load
import tkinter as tk
from tkinter import filedialog



def process_and_save_images_for_test(root_dir, x_axis_path, y_axis_path):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    X = np.array(imgs)
    Y = np.array(labels)

    with open(x_axis_path, mode='wb') as f:
        dump(X, f)

    with open(y_axis_path, mode='wb') as f:
        dump(Y, f)

    return X, Y


def load_single_image(img_path):
    img = preprocess_img(io.imread(img_path))
    return np.array([img], dtype='float32')


if __name__ == "__main__":
    imgs = []
    labels = []
    root_dir = 'Testing/'
    x_axis_filename = "X.bin"
    y_axis_filename = "Y.bin"
    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

    if os.path.exists(x_axis_filename) and os.path.exists(y_axis_filename):
        with open(x_axis_filename, mode="rb") as f:
            X = load(f)
        with open(y_axis_filename, mode="rb") as f:
            Y = load(f)
    else:
        X, Y = process_and_save_images_for_test(root_dir, x_axis_filename, y_axis_filename)

    with open("model.json") as f:
        json_str = '\n'.join(f.readlines())
        model = model_from_json(json_str)

    model.load_weights('model.h5')
    root = tk.Tk()
    root.withdraw()
    while True:
        file_path = filedialog.askopenfilename()
        abc = model.predict(load_single_image(file_path))
        for c, p in enumerate(abc[0]):
            if p > 1/NUM_CLASSES:
                print("{c} : {p}".format(c=c, p=p))


def test_accuracy():
    y_pred = model.predict_classes(X)
    acc = np.sum(y_pred == Y) / np.size(y_pred)
    print("Test accuracy = {}".format(acc))
