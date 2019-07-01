from sbs import SBS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
from r_forest import RandomForest


def plot_accuracy(subsets, scores):
    k_feat = [len(k) for k in subsets]
    plt.plot(k_feat, scores, marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Features number')
    plt.grid()
    plt.show()


def main():
    r_wine_path = os.path.join("data", "winequality-red.csv")
    w_wine_path = os.path.join("data", "winequality-white.csv")
    df_wine_red = pd.read_csv(r_wine_path, sep=';')
    df_wine_white = pd.read_csv(w_wine_path, sep=';')

    X_r, y_r = df_wine_red.iloc[:, :11], df_wine_red.iloc[:, 11]
    X_w, y_w = df_wine_white.iloc[:, :11], df_wine_white.iloc[:, 11]

    stdsc_r = StandardScaler()
    stdsc_w = StandardScaler()
    X_r_train_std = stdsc_r.fit_transform(X_r)
    X_w_train_std = stdsc_w.fit_transform(X_w)

    knn_r = KNeighborsClassifier(n_neighbors=5)
    knn_w = KNeighborsClassifier(n_neighbors=5)

    sbs_r = SBS(knn_r, k_features=1)
    sbs_w = SBS(knn_w, k_features=1)
    sbs_r.fit(X_r_train_std, y_r)
    sbs_w.fit(X_w_train_std, y_w)

    plot_accuracy(sbs_r.subsets_, sbs_r.scores_)
    plot_accuracy(sbs_w.subsets_, sbs_w.scores_)

    feat_labels = df_wine_white.columns[:11]

    RandomForest.f_importance(feat_labels, X_r, y_r)
    RandomForest.f_importance(feat_labels, X_w, y_w)


if __name__ == '__main__':
    main()