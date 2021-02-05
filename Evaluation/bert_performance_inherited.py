# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

import numpy as np
from sklearn import metrics

from Evaluation.performance_evaluation_product_classifier import PerformanceEvaluation
from Visualization.visualization import Visualization

# Overriding some methods from base Performance Evaluation class
class Performance(PerformanceEvaluation):
    def classification_report(self, targets, outputs, label_numbers, label_names):
        print("\nOverall Classification Report:")
        print(f"\n {metrics.classification_report(targets.argmax(axis=1), outputs.argmax(axis=1), labels = label_numbers, target_names = label_names)}")

    def compute_confusion_matrix(self, targets, outputs, label_numbers, label_names):
        targets = np.asarray(targets)
        confusion_mtx = metrics.confusion_matrix(targets.argmax(axis=1), outputs.argmax(axis=1), labels = label_numbers)
        Visualization.print_confusion_matrix(confusion_mtx, label_names)
