import joblib
from yolov11_model.model_predict import predict_image
from classifiers.feature_extraction import extract_features
from PIL import Image
import numpy as np

saved_cls = {
    'Bayes Classifier': 'classifiers/saved/bayes_classifier.pkl',
    'Decision Tree Classifier': 'classifiers/saved/decision_tree_classifier.pkl',
    'Random Forest Classifier': 'classifiers/saved/random_forest_classifier.pkl',
    'Neural Network Classifier': 'classifiers/saved/neural_network_model.pkl'
}

types = {
    0: 'benign',
    1: 'malignant'
}


def predict_image_info(image_path):
    image = Image.open(image_path)

    predict_yolo_model_info = predict_image(image_path)

    print(predict_yolo_model_info)

    coordinates = predict_yolo_model_info[0]["coordinates"]
    x_min, y_min, x_max, y_max = map(int, coordinates[0])

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image_np = np.array(cropped_image)
    data_extracted = extract_features(cropped_image_np)

    for classifier_name, classifier_path in saved_cls.items():
        load_cls = joblib.load(classifier_path)
        predicted_class = load_cls.predict([data_extracted])[0]
        print(f'{classifier_name}: {types[int(predicted_class)]}')


if __name__ == '__main__':
    predict_image_info('dataset/train/images/benign--108-_jpg.rf.e9bc8efdb56b660a1fc3bedc13246680.jpg')
