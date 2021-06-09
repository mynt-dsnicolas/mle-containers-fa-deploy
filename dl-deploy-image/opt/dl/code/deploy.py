import torch as th

from flask import Flask, request, jsonify
from fastai.vision.all import *
from pathlib import Path

app = Flask(__name__)

def load_model():
    path = Path('/home/deploy/opt/dl/model/')
    print('Loading model...')
    model = load_learner(path/'export.pkl')
    print('Model successfully loaded!')

    return model

def read_image(request):
    print('Reading image...')
    imagefile = request.files.get('imagefile', '')
    imagefile.save('/home/deploy/opt/dl/pred/pred_image.jpg')
    print('Image successfully read!')
    image_path = '/home/deploy/opt/dl/pred/pred_image.jpg'
    return image_path

def predict(img, model):
    print('Prediction in progress...')

    pred, pred_idx, probs = model.predict(img)
    return pred

@app.route('/invocations', methods=["POST"])
def main():
    model = load_model()
    img = read_image(request)
    pred = predict(img, model)

    response = {'prediction': str(pred)}
    print(f'prediction: {str(pred)}')
    return jsonify(response)

if __name__ == "__main__":
    app.run()