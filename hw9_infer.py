from io import BytesIO
from urllib import request

import numpy as np
import onnxruntime as rt
import torchvision.transforms as transforms
from PIL import Image

session = rt.InferenceSession("hair_classifier_empty.onnx")
train_transforms = transforms.Compose(
    [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet normalization
    ]
)


def predict():
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    test_data = train_transforms(
        download_image(
            "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
        )
    )
    test_data_np = test_data.numpy()  # (3, 200, 200)
    test_data_batch = np.expand_dims(test_data_np, axis=0).astype(
        np.float32
    )  # (1, 3, 200, 200)

    # run inference
    result = session.run([label_name], {input_name: test_data_batch})

    print(f"raw output: {result}")
    print(f"Prediction: {result[0][0][0]:.4f}")


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


if __name__ == "__main__":
    predict()
