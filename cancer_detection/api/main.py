from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, FastAPI
from pymongo import MongoClient
from typing import Any, Dict
from image_classifier.cnn import ImageCLFNet
import numpy as np
import json
from sklearn import preprocessing
import torch.nn.functional as F

import torch


import os

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model_path = '/Users/reshavabraham/personal_work/skin-cancer-detection/notebooks/cnn-model.pt'
model = ImageCLFNet()
model.load_state_dict(torch.load(model_path))
model.eval()


def test(test_loader, le):
    train_losses = []
    train_counter = []
    test_losses = []
    # test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test_loss = 0
    correct = 0
    predictions = []
    actual = []
    with torch.no_grad():
        for data, target in test_loader:
            try:
                data = data.view(64, 3, 75, 56).float()
            except:
                continue
            output = model(data)
            target = torch.tensor(le.transform(target))
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
            predictions.append(pred)
            actual.append(target)
            
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),acc))
    return test_loss, correct, len(test_loader.dataset), acc

@app.get("/")
async def main():
    return {"message": "Hello"}


@app.get("/getTestSetAcc")
async def get_test_acc():
    X_test = np.load("../../numpy-dump/test.npy", allow_pickle=True)
    # X_test = np.load("test.npy", allow_pickle=True)
    y_test = np.load("../../numpy-dump/test-labels.npy", allow_pickle=True)
    test_set = list(zip(X_test, y_test))

    labels_map = np.asarray(list(set(y_test))).reshape(-1, 1)

    le = preprocessing.LabelEncoder()
    le.fit(labels_map)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    test_loss, correct, d_len, acc = test(test_loader, le)
    return {"message": [test_loss, correct, d_len, acc]}