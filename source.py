import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

print("Loading MNIST...")
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
X_train = X_train_raw.reshape(-1, 784) / 255.0
X_test  = X_test_raw.reshape(-1, 784)  / 255.0

print("Training MLP...")
model = MLPClassifier(hidden_layer_sizes=(1024,512, 256, 128), max_iter=5, verbose=True)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

def predict(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=float)
    if arr.mean() > 128:
        arr = 255.0 - arr
    arr = np.where(arr > 30, arr, 0.0)
    rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        arr = arr[r0:r1+1, c0:c1+1]
    pad = max(arr.shape) // 4
    arr = np.pad(arr, pad, constant_values=0)
    arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((28, 28)), dtype=float) / 255.0
    plt.imshow(arr.reshape(28, 28), cmap="gray")
    plt.title("What the model sees"); plt.show()
    probs = model.predict_proba([arr.flatten()])[0]
    pred  = np.argmax(probs)
    print(f"Predicted: {pred}")


y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
digits = list(range(10))
tp = [report[str(d)]['precision'] for d in digits]
fp = [1 - report[str(d)]['precision'] for d in digits]
x = np.arange(10)
width = 0.35
plt.bar(x - width/2, tp, width, label="True Positive",  color="steelblue")
plt.bar(x + width/2, fp, width, label="False Positive", color="red")
plt.xticks(x)
plt.xlabel("Digit")
plt.ylabel("Rate")
plt.title("True Positive vs False Positive Rate per Digit")
plt.legend()
plt.show()

import os
folder = os.path.dirname(os.path.abspath(__file__))

for x in range(2,6):
    predict(os.path.join(folder, f'{x}.png'))



