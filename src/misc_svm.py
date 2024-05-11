import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import svm
from sklearn.model_selection import cross_val_score

def load_csv(data_path):
    return pd.read_csv(data_path)

def check_null_values(data):
    return data.isnull().sum()

def drop(data, *column_drop, axis = None):
    df1 =  data.drop(list(column_drop), axis = axis)
    return df1  

def pair_plot(df1, text_size=12, figsize=(15, 15)):
    # Set the font size for axis labels
    sns.set(font_scale=1.2, rc={"axes.labelsize": text_size, "axes.titlesize": text_size, "xtick.labelsize": text_size, "ytick.labelsize": text_size})

    # Set the figure size
    plt.figure(figsize=figsize)

    # Create the pair plot
    sns.pairplot(df1)

    # Show the plot
    plt.show()

def independent_variable(df1, *columns, axis = None):
    df2 = df1.drop(list(columns), axis = axis)
    return df2

def create_svm(kernel = None, gamma = None, C = None):
    svc_create = OneVsRestClassifier(SVC(kernel = kernel, gamma = gamma, C = C))
    return svc_create


def fit_svm(svc_create , X_train_pca, Y_train):
    svc = svc_create.fit(X_train_pca, Y_train)
    return svc

def pred(svc, X_test_pca):
    y_pred = svc.predict(X_test_pca)
    return y_pred

def accuracy_scoring(y_pred, Y_test):
    prediction = accuracy_score(y_pred, Y_test)
    return prediction

def confusion_matrixes(y_pred, Y_test, cmap=None, annot=None, fmt=None, textsize=None, colorbar=True):
    cm = confusion_matrix(y_pred, Y_test)
    cm_matrix = pd.DataFrame(data=cm, columns=["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7"],
                             index=["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7"])
    heatmap = sns.heatmap(cm_matrix, cmap=cmap, annot=annot, fmt=fmt, annot_kws={'size': textsize})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=textsize)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=textsize)
    heatmap.set_xlabel("Actual Value", fontsize=textsize)
    heatmap.set_ylabel("Predicted Value", fontsize=textsize)

    if colorbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=textsize)

    heatmap.tick_params(axis='both', labelsize=textsize)  # Adjust tick parameters for both axes

    plt.show()

def create_classification_report(y_pred, Y_test):
    classification_repo = classification_report(Y_test, y_pred)
    return classification_repo

def visualize_decision_boundaries(svc, X_train, Y_train, title="Decision Boundaries"):
    # Convert X_train to numpy array if it's not already
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)

    # Create a meshgrid of feature values
    h = 0.01  # Step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Make predictions on the meshgrid points
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predictions into the meshgrid shape
    Z = Z.reshape(xx.shape)

    # Set figure size
    plt.figure(figsize=(3, 3))

    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Paired)

    # Plot training data
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='k', cmap=plt.cm.Paired)
    
    # Add legend with class labels
    classes = np.unique(Y_train)
    legend_labels = [f"ID{cls}" for cls in classes]
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, fontsize=12, title="Individual Sika deer")
    plt.setp(legend.get_title(), fontsize=12)  # Set legend title fontsize
    
    # Adjust contour line properties
    plt.contour(xx, yy, Z, colors='k', linewidths=2)

    plt.xlabel('PCA1', fontsize=12)
    plt.ylabel('PCA2', fontsize=12)
    #plt.title(title, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Move legend outside the plot
    plt.subplots_adjust(right=0.85)  # Adjust the right margin
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, fontsize=12, title="Individual Sika deer", loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


def pca_analysis_point(X_train, X_test, Y_train):
    pca = PCA(n_components=2)
    scaler_pca = StandardScaler()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_train = scaler_pca.fit_transform(X_train)
    X_test = scaler_pca.transform(X_test)

    plt.figure(figsize=(3, 3))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap="magma")
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')

    # Annotate PCA clusters
    #for i, label in enumerate(Y_train):
    #    plt.text(X_train[i, 0], X_train[i, 1], str(label), fontsize=10, color='black')

    plt.show()

def plot_roc_curve(y_test, y_pred, n_classes):
    # Make sure that classes start from 1
    y_test = y_test
    y_pred = y_pred

    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=np.arange(1, n_classes + 1))
    
    # Binarize the predicted labels
    y_pred_bin = label_binarize(y_pred, classes=np.arange(1, n_classes + 1))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(3, 3))  # Adjust the figure size as needed
    colors = ['#ff0000', '#ff8000', '#00ff00', '#0040ff', '#8000ff', '#ff00ff', '#ff0080']  # You can customize the colors
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ID{i+1} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 12)  # Place legend outside the graph
    plt.show()

def optimal_gamma_c_heatmap(X_train, X_test, Y_train, Y_test, svc_create):
    # Define the range of gamma and C values to be explored
    gamma_values = np.logspace(-3, 3, 7)
    C_values = np.logspace(-3, 3, 7)
    
    # Create a parameter grid for the grid search
    param_grid = {'estimator__gamma': gamma_values, 'estimator__C': C_values}
    
    # Create an SVM model
    svm = svc_create
    
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    
    # Extract and reshape the results for visualization
    results = grid_search.cv_results_
    scores = np.array(results['mean_test_score']).reshape(len(gamma_values), len(C_values))

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_gamma = best_params['estimator__gamma']
    best_C = best_params['estimator__C']

    # Print the best hyperparameters
    print(f"Best parameters - Gamma: {best_gamma}, C: {best_C}")

    # Print the top five combinations of hyperparameters
    print("\nTop five combinations of hyperparameters:")
    sorted_indices = np.argsort(-results['mean_test_score'])[:5]
    for i, index in enumerate(sorted_indices):
        gamma_index = index // len(C_values)
        C_index = index % len(C_values)
        gamma_val = gamma_values[gamma_index]
        C_val = C_values[C_index]
        print(f"{i+1}. Gamma: {gamma_val}, C: {C_val}, Mean Test Score: {results['mean_test_score'][index]}")

    # Plot a heatmap of mean test scores
    plt.figure(figsize=(8, 8))
    
    # Increase the fontsize of annotations using annot_kws
    heatmap = sns.heatmap(scores, annot=True, fmt='.2f', annot_kws={"size": 20},
                          xticklabels=["{:.2f}".format(val) for val in C_values],
                          yticklabels=["{:.2f}".format(val) for val in gamma_values], cmap='magma')
    
    # Set labels for axes
    plt.xlabel('C', fontsize=14)
    plt.ylabel('Gamma', fontsize=14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    # Add colorbar with labels
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Mean Test Score', rotation=270, labelpad=15, fontsize=14)
    cbar.ax.tick_params(labelsize = 14)

    # Show the plot
    plt.show()

def optimal_gamma_c_line(X_train, X_test, Y_train, Y_test, svc_create):
    gamma_values = np.logspace(-3, 3, 7)
    C_values = np.logspace(-3, 3, 7)
    param_grid = {'estimator__gamma': gamma_values, 'estimator__C': C_values}
    svm = svc_create
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    results = grid_search.cv_results_

    # Extract scores and reshape them
    scores = np.array(results['mean_test_score']).reshape(len(gamma_values), len(C_values))

    # Plotting line graphs
    plt.figure(figsize=(4, 4))
    
    # Plot lines for each gamma value
    for i, gamma_val in enumerate(gamma_values):
        plt.plot(C_values, scores[i, :], label=f'Gamma={gamma_val:.3f}', marker='o')

    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. C for different Gamma values')
    #plt.xticks(C_values, ["{:.2f}".format(val) for val in C_values])  # Update x-axis ticks
    
    # Place legend outside the graph
    plt.legend(title='Gamma', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()

def perform_kfold_cross_validation(X_train, Y_train, svc_create, n_splits=7):
    # Create a StratifiedKFold object for stratified k-fold cross-validation
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform k-fold cross-validation and get the accuracy scores
    scores = cross_val_score(svc_create, X_train, Y_train, cv=stratified_kfold, scoring='accuracy')

    # Print the accuracy scores for each fold
    for fold, score in enumerate(scores, start=1):
        print(f'Fold {fold}: Accuracy = {score:.4f}')

    # Print the mean and standard deviation of the accuracy scores
    print(f'\nMean Accuracy: {np.mean(scores):.4f}')
    print(f'Standard Deviation: {np.std(scores):.4f}')    

def plot_kernel_performance(X_train, Y_train, kernels):
    mean_accuracies = []

    for kernel in kernels:
        clf = svm.SVC(kernel=kernel)
        scores = cross_val_score(clf, X_train, Y_train, cv=10)  # 10-fold cross-validation
        mean_accuracy = scores.mean()
        mean_accuracies.append(mean_accuracy)

        print(f"Kernel: {kernel}, Mean Accuracy: {mean_accuracy}")

    plt.figure(figsize=(4, 4))
    bar_width = 0.2  # Adjust the width of the bars
    space = 0.05  # Adjust the space between bars
    color = '#51829B'  # Use a single color for all bars

    x_positions = np.arange(len(kernels)) * (bar_width + space)  # Adjust positions of bars with space

    bars = plt.bar(x_positions, mean_accuracies, color=color, width=bar_width)
    plt.xlabel('Kernels', fontsize=14)
    plt.ylabel('Mean Accuracy', fontsize=14)
    plt.ylim([0, 1])  # Set y-axis limit between 0 and 1 for accuracy
    plt.yticks(fontsize=14)
    plt.xticks(x_positions, kernels, fontsize=14, rotation='horizontal')  # Rotate x-axis tick labels vertically

    # Add percentage labels on top of each bar
    #for bar, accuracy in zip(bars, mean_accuracies):
    #    height = bar.get_height()
    #    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{accuracy*100:.2f}%', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
