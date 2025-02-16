import joblib
from yolov11_model.model_predict import predict_image
from classifiers.feature_extraction import extract_features
from PIL import Image
import numpy as np

saved_cls = {
    'Naive-Bayes': 'classifiers/saved/bayes_classifier.pkl',
    'Decision Tree': 'classifiers/saved/decision_tree_classifier.pkl',
    'Random Forest': 'classifiers/saved/random_forest_classifier.pkl',
    'Neural Network': 'classifiers/saved/neural_network_model.pkl'
}

types = {
    0: 'benign',
    1: 'malignant'
}


def predict_image_info(image_path):
    classifiers_predictions = {'YOLOv11': None,
                               'Naive-Bayes': None,
                               'Decision Tree': None,
                               'Random Forest': None,
                               'Neural Network': None
                               }

    image = Image.open(image_path)

    predict_yolo_model_info = predict_image(image_path)
    print(predict_yolo_model_info)
    if not predict_yolo_model_info:
        return classifiers_predictions

    print(f"YOLOv11 Model Prediction Results: {types[predict_yolo_model_info[0]['label']]}. Confidence: {predict_yolo_model_info[0]['confidence']:2f}")

    coordinates = predict_yolo_model_info[0]["coordinates"]
    x_min, y_min, x_max, y_max = map(int, coordinates[0])

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image_np = np.array(cropped_image)
    data_extracted = extract_features(cropped_image_np)

    classifiers_predictions['YOLOv11'] = types[predict_yolo_model_info[0]['label']]
    for classifier_name, classifier_path in saved_cls.items():
        load_cls = joblib.load(classifier_path)
        predicted_class = load_cls.predict([data_extracted])[0]
        # print(f'{classifier_name} Classifier Prediction: {types[int(predicted_class)]}')
        classifiers_predictions[classifier_name] = types[int(predicted_class)]

    return classifiers_predictions


if __name__ == '__main__':
    for cls_name, cls_pred in predict_image_info('assets/malignant.jpg').items():
        print(f'{cls_name} Classifier Prediction: {cls_pred}')
    for cls_name, cls_pred in predict_image_info('assets/benign.png').items():
        print(f'{cls_name} Classifier Prediction: {cls_pred}')
