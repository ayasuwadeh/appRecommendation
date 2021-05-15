import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToJson


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'google-key.json'


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # jsonObj = MessageToJson(texts.description)
    # print(jsonObj)
    i = 0
    back=''
    for text in texts:
        if i == 0:
            i = i+1
            back += text.description
            print('\n"{}"'.format(text.description))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return back
