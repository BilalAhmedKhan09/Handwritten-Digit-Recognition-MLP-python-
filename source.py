import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier
from scipy.ndimage import rotate, zoom
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


print("Loading MNIST...")
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
X_train = X_train_raw.reshape(-1, 784) / 255.0
X_test  = X_test_raw.reshape(-1, 784)  / 255.0

def augment(X, y, factor=2):
    aug_X, aug_y = [X], [y]
    for _ in range(factor):
        new_X = []
        for img in X:
            img28 = img.reshape(28, 28)
            img28 = rotate(img28, angle=np.random.uniform(-15, 15), reshape=False)
            img28 = np.roll(img28, np.random.randint(-3, 3), axis=0)
            img28 = np.roll(img28, np.random.randint(-3, 3), axis=1)
            new_X.append(np.clip(img28.flatten(), 0, 1))
        aug_X.append(np.array(new_X))
        aug_y.append(y)
    return np.vstack(aug_X), np.concatenate(aug_y)

X_train_aug, y_train_aug = augment(X_train, y_train, factor=2)
print("Training MLP...")
model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=2, verbose=True)
model.fit(X_train_aug, y_train_aug)
print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
pickle.dump(model, open("digit_mlp.pkl", "wb"))


def predict(path):
    model = pickle.load(open("digit_mlp.pkl", "rb"))
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
plt.bar(x - width/2, tp, width, label="True Positive Rate",  color="steelblue")
plt.bar(x + width/2, fp, width, label="False Positive Rate", color="coral")
plt.xticks(x)
plt.xlabel("Digit")
plt.ylabel("Rate")
plt.title("True Positive vs False Positive Rate per Digit")
plt.legend()
plt.show()

for x in range(2,6):
    predict(f'{x}.png') 

