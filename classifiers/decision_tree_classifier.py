from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
import joblib
from naive_bayes_model import read_file


if __name__ == '__main__':
    dataset = read_file('data_features.csv')

    train_set = dataset[:int(0.8 * len(dataset))]
    train_x = [row[1:] for row in train_set]
    train_y = [row[0] for row in train_set]

    test_set = dataset[int(0.8 * len(dataset)):]
    test_x = [row[1:] for row in test_set]
    test_y = [row[0] for row in test_set]

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=0.04)
    classifier.fit(train_x, train_y)
    joblib.dump(classifier, 'saved/decision_tree_classifier.pkl')

    # REPORT
    y_pred = classifier.predict(test_x)
    cr = classification_report(test_y, y_pred)
    print(cr)

    print(f'Depth: {classifier.get_depth()}')
    print(f'Number of leaves: {classifier.get_n_leaves()}')
    features_importance = list(classifier.feature_importances_)
    print(features_importance)
    most_important_feature = features_importance.index(max(features_importance))
    print(f'Most important feature: {most_important_feature}')

    least_important_feature = features_importance.index(min(features_importance))
    print(f'Least important feature: {least_important_feature}')

    accuracy = accuracy_score(test_y, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    average_precision = average_precision_score(test_y, y_pred, average='macro')
    print(f'Average Precision: {average_precision:.2f}')


