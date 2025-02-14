import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
import joblib


def read_file(file_name):
    with open(file_name) as doc:
        csv_reader = csv.reader(doc, delimiter=',')
        dataset = list(csv_reader)

    dataset_v2 = []
    for row in dataset:
        row_v2 = [float(el) for el in row]
        dataset_v2.append(row_v2)

    return dataset_v2


if __name__ == '__main__':
    dataset = read_file('data_features.csv')

    train_set = dataset[:int(0.8 * len(dataset))]
    train_x = [row[1:] for row in train_set]
    train_y = [row[0] for row in train_set]

    test_set = dataset[int(0.8 * len(dataset)):]
    test_x = [row[1:] for row in test_set]
    test_y = [row[0] for row in test_set]

    # scaler = MinMaxScaler()
    # train_x = scaler.fit_transform(train_x)
    # test_x = scaler.transform(test_x)

    classifier = GaussianNB()
    classifier.fit(train_x, train_y)

    joblib.dump(classifier, 'saved/bayes_classifier.pkl')

    y_pred = classifier.predict(test_x)
    cr = classification_report(test_y, y_pred)
    print(cr)

    accuracy = accuracy_score(test_y, y_pred)
    print(f'Accuracy: {accuracy}')

    average_precision = average_precision_score(test_y, y_pred, average='macro')
    print(f'Average Precision: {average_precision:.2f}')