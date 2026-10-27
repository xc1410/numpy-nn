import os
import urllib.request

FILES = {
    "train-images-idx3-ubyte.gz": 9912422,
    "train-labels-idx1-ubyte.gz": 28881,
    "t10k-images-idx3-ubyte.gz": 1648877,
    "t10k-labels-idx1-ubyte.gz": 4542,
}
SOURCES = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

os.makedirs("data", exist_ok=True)
for name, size in FILES.items():
    path = os.path.join("data", name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print("已存在", name)
        continue
    for source in SOURCES:
        try:
            with urllib.request.urlopen(source + name, timeout=60) as response:
                content = response.read()
        except Exception as error:
            print("失败", source, type(error).__name__)
            continue
        if len(content) != size:
            print("大小不对", source, len(content))
            continue
        with open(path, "wb") as file:
            file.write(content)
        print("下好了", name, "来自", source)
        break
    else:
        print("三个源都不行：", name)