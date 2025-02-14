import joblib
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.neural_network import MLPClassifier
import csv
import numpy as np


def read_file(file_name):
    with open(file_name, 'r') as doc:
        csv_reader = csv.reader(doc, delimiter=',')
        dataset = [list(map(float, row)) for row in csv_reader]
    return dataset


def divide_sets(dataset, train_ratio=0.7, val_ratio=0.1):
    class_m = [x for x in dataset if x[0] == 1]
    class_b = [x for x in dataset if x[0] == 0]

    train_idx_m = int(len(class_m) * train_ratio)
    val_idx_m = train_idx_m + int(len(class_m) * val_ratio)

    train_idx_b = int(len(class_b) * train_ratio)
    val_idx_b = train_idx_b + int(len(class_b) * val_ratio)

    train_set = class_m[:train_idx_m] + class_b[:train_idx_b]
    val_set = class_m[train_idx_m:val_idx_m] + class_b[train_idx_b:val_idx_b]
    test_set = class_m[val_idx_m:] + class_b[val_idx_b:]

    return train_set, val_set, test_set


def extract_features_labels(data_set):
    features = np.array([x[1:] for x in data_set])
    labels = np.array([x[0] for x in data_set])
    return features, labels


def train_and_evaluate(models, train_x, train_y, val_x, val_y):
    best_model = None
    max_acc = 0

    for i, model in enumerate(models):
        model.fit(train_x, train_y)
        val_predictions = model.predict(val_x)
        val_acc = accuracy_score(val_y, val_predictions)
        print(f'Classifier {i + 1} val acc: {val_acc:.2f}')

        if val_acc > max_acc:
            max_acc = val_acc
            best_model = model

    return best_model


if __name__ == '__main__':
    dataset = read_file("data_features.csv")
    train_set, val_set, test_set = divide_sets(dataset)

    train_x, train_y = extract_features_labels(train_set)
    val_x, val_y = extract_features_labels(val_set)
    test_x, test_y = extract_features_labels(test_set)

    # scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # val_x = scaler.transform(val_x)
    # test_x = scaler.transform(test_x)

    models = [
        MLPClassifier(hidden_layer_sizes=(5,), activation='relu', learning_rate_init=0.001, max_iter=500,
                      random_state=0),
        MLPClassifier(hidden_layer_sizes=(10,), activation='relu', learning_rate_init=0.001, max_iter=500,
                      random_state=0),
        MLPClassifier(hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, max_iter=500,
                      random_state=0)
    ]

    final_classifier = train_and_evaluate(models, train_x, train_y, val_x, val_y)
    joblib.dump(final_classifier, 'saved/neural_network_model.pkl')

    test_predictions = final_classifier.predict(test_x)
    test_acc = accuracy_score(test_y, test_predictions)
    print(f'Test accuracy {test_acc:.2f}')

    print("Report")
    print(classification_report(test_y, test_predictions))

    avg_precision = average_precision_score(test_y, test_predictions, average='macro')
    print(f'Average Precision {avg_precision:.2f}')
