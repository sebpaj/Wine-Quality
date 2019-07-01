from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


class RandomForest:

    @staticmethod
    def f_importance(feat_labels, X_train, y_train):
        forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
        plt.title("Features importance")
        plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
        plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        plt.show()