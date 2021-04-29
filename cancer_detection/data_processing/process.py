import numpy as np
import time


start = time.time()

X_train = np.load("train.npy", allow_pickle=True)
X_train = np.divide(X_train, 255)

mean, std = X_train.mean(), X_train.std()

X_train = (X_train - mean) / std


X_test = np.load("test.npy", allow_pickle=True)
X_test = np.divide(X_test, 255)

mean, std = X_test.mean(), X_test.std()

X_test = (X_test - mean) / std

end = start - time.time()


np.save("normalized-train-images", X_train)
np.save("normalized-test-images", X_test)
print("end", end)