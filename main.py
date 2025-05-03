import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def load_images(folder, num_images_per_category):
    data = []
    targets = []
    categories = sorted(p.stem for p in folder.iterdir() if p.is_dir())
    for label, category in enumerate(tqdm(categories, desc='Loading Images')):
        img_paths = sorted((folder / category).iterdir())
        assert num_images_per_category <= len(img_paths)
        for img_path in img_paths[: num_images_per_category]:
            img = Image.open(img_path).convert('L')
            img = img.resize((64, 64))
            data.append(np.array(img).flatten())
            targets.append(label)

    return np.array(data), np.array(targets), categories


def preprocess(inputs, targets):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.10, random_state=0,
    )
    inputs_train = inputs_train / np.float32(255.)
    inputs_test = inputs_test / np.float32(255.)
    return inputs_train, inputs_test, targets_train, targets_test


def confusion_mat(target, predictions, labels, title, num_imgs):
    cm = ConfusionMatrixDisplay(confusion_matrix(target, predictions), display_labels=labels)
    fig, ax = plt.subplots()
    cm.plot(ax=ax)
    ax.set_title(title)
    plt.savefig(f'{title.lower().replace(" ", "_")}--num_imgs-{num_imgs}--confusion-mat')


def experiment_multiple_models(model_cls, hyperparam_name, hyperparam_vals, inputs_train, inputs_test, targets_train, targets_test):
    accuracy = []
    progress_bar = tqdm(
        hyperparam_vals,
        desc=(
            f'Experimenting with {hyperparam_name} '
            f'for model `{model_cls.__name__}`'
        )
    )
    for hv in progress_bar:
        model = model_cls(**{hyperparam_name: hv}, random_state=0)
        model.fit(inputs_train, targets_train)
        preds = model.predict(inputs_test)
        acc = accuracy_score(targets_test, preds)
        accuracy.append(acc)
    plt.figure(figsize=(10, 6))
    if all(isinstance(v, tuple) for v in hyperparam_vals):
        x_vals = [len(t) for t in hyperparam_vals]
        title = f'Performance vs {hyperparam_name} for {model_cls.__name__}'
    else:
        x_vals = hyperparam_vals
        title = f'Performance vs {hyperparam_name} for {model_cls.__name__}'
    plt.plot(x_vals, accuracy)
    plt.xlabel(hyperparam_name)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig((
        f'{model_cls.__name__}--num_imgs-'
        f'{len(inputs_train) + len(inputs_test)}'
        f'--{hyperparam_name}-vs-accuracy'
    ).lower())


def experiment_single_models(inputs_train, inputs_test, targets_train, targets_test, labels):
    mlp = MLPClassifier(hidden_layer_sizes=(25, 50, 10), batch_size=32, learning_rate_init=0.01, max_iter=200, random_state=0)

    rf = RandomForestClassifier(n_estimators=100, random_state=0)

    mlp.fit(inputs_train, targets_train)
    mlp_preds = mlp.predict(inputs_test)
    mlp_acc = accuracy_score(targets_test, mlp_preds)

    rf.fit(inputs_train, targets_train)
    rf_preds = rf.predict(inputs_test)
    rf_acc = accuracy_score(targets_test, rf_preds)

    print(f'MLP Accuracy: {mlp_acc:.2%}')
    print(f'Random Forest Accuracy: {rf_acc:.2%}')

    confusion_mat(
        targets_test, mlp_preds, labels=labels, title=mlp.__class__.__name__,
        len(inputs_train) + len(inputs_test)
    )
    confusion_mat(
        targets_test, rf_preds, labels=labels, title=rf.__class__.__name__,
        len(inputs_train) + len(inputs_test)
    )


def main():
    for num_images in [50, 100, 200]:
        print(f"\n=== Running experiments with {num_images} images per category ===\n")
        inputs, targets, labels = load_images(
            # We are only grabbing data from the train set and splitting it
            # into train and test. We are ignoring the `asl_alphabet_test`
            # folder
            Path('Sign_Language_Dataset') / 'asl_alphabet_train' / 'asl_alphabet_train',
            num_images_per_category=num_images
        )
        inputs_train, inputs_test, targets_train, targets_test = preprocess(inputs, targets)
        experiment_single_models(inputs_train, inputs_test, targets_train, targets_test, labels)
        experiment_multiple_models(
            RandomForestClassifier, 'n_estimators', [50, 100, 200, 300, 400],
            inputs_train, inputs_test, targets_train, targets_test
        )
        experiment_multiple_models(
            MLPClassifier, 'hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 100), (100, 50, 25)],
            inputs_train, inputs_test, targets_train, targets_test
        )
        experiment_multiple_models(
            MLPClassifier, 'batch_size', [16, 32, 64, 128],
            inputs_train, inputs_test, targets_train, targets_test
        )

        experiment_multiple_models(
            MLPClassifier, 'learning_rate_init', [0.0001, 0.001, 0.01, 0.1],
            inputs_train, inputs_test, targets_train, targets_test
        )
        experiment_multiple_models(
            MLPClassifier, 'max_iter', [50, 100, 200, 300],
            inputs_train, inputs_test, targets_train, targets_test
        )


if __name__ == '__main__':
    main()
