# src/eval.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, threshold=0.5):
    preds = model.predict(X_test)
    probs = preds[:,1]
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    return {'confusion_matrix': cm, 'report': report, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
