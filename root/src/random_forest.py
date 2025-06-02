from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os

def run_random_forest(X,X_train, X_test, y_train, y_test, class_labels,result_dir):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🌲 Random Forest")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred)) 
    print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))
    print("F1 (weighted):", f1_score(y_test, y_pred, average='weighted'))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.imshow(cm, cmap='Greens')
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, "RF_Confusion_matrix.png"))
    plt.close

    # Feature Importances (her durumda gösterilebilir)
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color='seagreen')
    plt.xlabel("Önem Skoru")
    plt.title("Features Importance - Rock Type")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "RF_Features_Importance.png"))
    plt.close()

    # ROC (if binary)
    if len(np.unique(y_test)) == 2:
        y_test_bin = label_binarize(y_test, classes=class_labels)
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("ROC Curve - Random Forest")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, "RF_roc_curve.png"))
        plt.close
    return model
