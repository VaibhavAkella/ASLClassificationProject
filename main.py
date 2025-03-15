from sklearn.neural_network import MLPClassifier
import numpy as np
from PIL import Image

from pathlib import Path

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

def main():
    inputs, targets = load_images(Path('animals'))
    classifier = MLPClassifier(random_state=1)
    print(f'Shape of input data array:  {inputs.shape}')
    print(f'Shape of output data array: {targets.shape}')
    test_size = 100
    classifier.fit(inputs[test_size:], targets[test_size:])
    results = classifier.predict(inputs[:test_size])

    print("Predictions:", results)
    print("Actual:", targets[:test_size])
    print("% of wrong predictions:",100*sum(results)/test_size)

if __name__ == '__main__':
    main()
