# 📋 粘贴：读 MNIST 的四个 .gz 文件，图片拉成 784 维、缩放到 0–1
import gzip
import numpy as np


def read_images(path):
    with gzip.open(path, "rb") as file:
        raw = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
    return raw.reshape(-1, 784).astype(np.float64) / 255.0


def read_labels(path):
    with gzip.open(path, "rb") as file:
        return np.frombuffer(file.read(), dtype=np.uint8, offset=8).astype(np.int64)


def load_mnist():
    train_images = read_images("data/train-images-idx3-ubyte.gz")
    train_labels = read_labels("data/train-labels-idx1-ubyte.gz")
    test_images = read_images("data/t10k-images-idx3-ubyte.gz")
    test_labels = read_labels("data/t10k-labels-idx1-ubyte.gz")
    return train_images, train_labels, test_images, test_labels