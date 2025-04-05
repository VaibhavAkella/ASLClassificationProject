import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def load_images(folder):
    data = []
    targets = []

    for label, category in enumerate(['cat', 'dog']):
        for img_path in sorted((folder / category).iterdir()):
            img = Image.open(img_path).convert('L')
            img = img.resize((64, 64))
            data.append(np.array(img).flatten())
            targets.append(label)

    return np.array(data), np.array(targets)


def preprocess(inputs, targets):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs.astype(np.float32) // 255.0, targets, test_size=0.10, random_state=0
    )
    return inputs_train, inputs_test, targets_train, targets_test


def confusion_mat(target, predictions, labels, title):
    cm = ConfusionMatrixDisplay(confusion_matrix(target, predictions), display_labels=labels)
    fig, ax = plt.subplots()
    cm.plot(ax=ax)
    ax.set_title(title)
    plt.show()


def main():
    inputs, targets = load_images(Path('animals'))
    inputs_train, inputs_test, targets_train, targets_test = preprocess(inputs, targets)

    mlp = MLPClassifier(hidden_layer_sizes=(25, 50, 10), batch_size=32, learning_rate_init=0.01, max_iter=200, random_state=0)

    rf = RandomForestClassifier(n_estimators=100, random_state=0)

    mlp.fit(inputs_train, targets_train)
    mlp_preds = mlp.predict(inputs_test)
    mlp_acc = accuracy_score(targets_test, mlp_preds)

    rf.fit(inputs_train, targets_train)
    rf_preds = rf.predict(inputs_test)
    rf_acc = accuracy_score(targets_test, rf_preds)

    print(f"MLP Accuracy: {mlp_acc:.2%}")
    print(f"Random Forest Accuracy: {rf_acc:.2%}")

    confusion_mat(targets_test, mlp_preds, labels=['Cat', 'Dog'], title="MLP Classifier")
    confusion_mat(targets_test, rf_preds, labels=['Cat', 'Dog'], title="Random Forest")


if __name__ == '__main__':
    main()
