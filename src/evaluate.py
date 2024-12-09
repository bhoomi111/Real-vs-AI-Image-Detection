from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    fpr, tpr, _ = roc_curve(y_true_labels, y_pred[:, 1])
    auc_score = auc(fpr, tpr)
    print(f"AUC: {auc_score}")
