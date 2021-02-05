from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

class PerformanceEvaluation():
    def __init__(self):
        self.average = ["micro", "macro", "weighted"]

    def overall_f1_score(self, targets, outputs, accuracy=None):
        print("\nComputing the networks performance:\n")
        f1_score = []
        for f1_type in self.average:
            f1_score.append(metrics.f1_score(targets, outputs, average=f1_type))
        
        print(f"\nOverall Accuracy = {accuracy}")
        print(f"\nOverall F1 Score (Micro) = {f1_score[0]}")
        print(f"\nOverall F1 Score (Macro) = {f1_score[1]}")
        print(f"\nOverall F1 Score (Weighted) = {f1_score[2]}")
        
    def classification_report(self, targets, outputs, label_numbers, label_names):
        print("\nOverall Classification Report:")
        print(f"\n {metrics.classification_report(targets, outputs, labels=label_numbers, target_names=label_names)}")

    def compute_confusion_matrix(self, targets, outputs, label_numbers):
        #TODO: Check if labels match targets
        confusion_mtx = metrics.confusion_matrix(targets, outputs, labels=label_numbers)
        return confusion_mtx

    def pairwise_cosine_similarity(self, feature_vector_1, feature_vector_2):
        cos_sim = cosine_similarity(feature_vector_1, feature_vector_2, dense_output = False)

        return cos_sim