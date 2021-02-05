# For dataframe
import pandas as pd

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_learning_curves

class Visualization:
    def __init__(self):
        self.plot_style = 'seaborn'

    # Useful to plot training curves for neural models models such as LSTM, CNN, BERT etc.
    def learning_plots(self, train_loss, validation_loss, fontsize=40):
        plt.style.use(self.plot_style)
        plt.suptitle('Loss Curves', fontsize=fontsize)
        plt.plot(train_loss, label= 'Training Loss')
        plt.plot(validation_loss, color= 'orange', label= 'Validation Loss')
        plt.legend()
        plt.xlabel('Epochs/Steps')
        plt.ylabel('Loss')
        plt.show()
    
    def print_confusion_matrix(self, confusion_matrix, class_names, figsize = (10,10), fontsize=16):
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        fig = plt.figure(figsize=figsize)

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar = True)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    # Useful to plot training curves for models such as SVM, Random Forest, Decision Trees, and much more.
    def plot_ml_training_curves(self, train_x, train_y, test_x, test_y, model_object):
        plot_learning_curves(train_x, train_y, test_x, test_y, model_object, scoring = 'accuracy', style = 'dark_background')
        
        plt.show()