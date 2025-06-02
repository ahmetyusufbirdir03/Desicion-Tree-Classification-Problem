from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os

def run_decision_tree(X, X_train, X_test, y_train, y_test, class_labels, result_dir):
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🌲 Decision Tree")
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))
    print("F1 (macro):", round(f1_score(y_test, y_pred, average='macro'), 4))
    print("F1 (weighted):", round(f1_score(y_test, y_pred, average='weighted'), 4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.title("Confusion Matrix - Decision Tree")
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "DT_Confusion_matrix.png"))
    plt.close()

    # ROC (if binary)
    if len(np.unique(y_test)) == 2:
        y_test_bin = label_binarize(y_test, classes=class_labels)
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("ROC Curve - Decision Tree")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "DT_Roc_curve.png"))
        plt.close()

    # Feature Importances (her durumda gösterilebilir)
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color='seagreen')
    plt.xlabel("Önem Skoru")
    plt.title("Features Importance - Rock Type")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "DT_Features_Importance.png"))
    plt.close()

    return model
