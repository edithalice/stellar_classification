import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from SciServer import CasJobs, SkyQuery, SciDrive, Authentication
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def login():
    '''Log in to SciServer'''
    return Authentication.login('******', '******')

def _group_target(subclass):
    '''Group target into classes.'''
    grouped_classes = {'Carbon': 'CarbonWD', 'A': 'A',
                       'B': 'OB', 'F': 'F', 'G': 'G',
                       'K': 'K', 'M': 'M', 'L': 'LT',
                       'O': 'OB', 'T': 'LT', 'WD': 'CarbonWD',
                       'CV': 'CV'}
    for k, v in grouped_classes.items():
        if k in subclass:
            return v
    return np.nan

def grouped_frame(df):
    '''Change target column to grouped value.'''
    df_new = df.copy()
    df_new.SUBCLASS = df_new.SUBCLASS.apply(_group_target)
    return df

def frame_from_drive(path, grouped=True, replace_9=False):
    '''Create DataFrame from SciDrive csv at given path.'''
    try:
        csv_url = SciDrive.publicUrl(f'metis_project_3/{path}.csv')
    except Exception as err:
        if str(err) == 'User token is not defined. First log into SciServer.':
            token = login()
            csv_url = SciDrive.publicUrl(f'metis_project_3/{path}.csv')
        else:
            raise err
    if replace_9:
        df = pd.read_csv(csv_url, index_col=0, na_values = '-9.999')
    else:
        df = pd.read_csv(csv_url, index_col=0)
    if grouped:
        df = grouped_frame(df)
    return df


def frame_from_url(csv_url, grouped=True, replace_9=False):
    '''Create DataFrame from SciDrive csv at given url.'''
    if replace_9:
        df = pd.read_csv(csv_url, index_col=0, na_values = '-9.999')
    else:
        df = pd.read_csv(csv_url, index_col=0)
    if grouped:
        df = grouped_frame(df)
    return df

def fit_score(classifier, X_train, y_train, X_test, y_test):
    '''
    Fit and score a classifier model for given data.
    Returns:
    - length of training time in seconds
    - length of fit time in seconds
    -  test_scores dict containing overall accuracy along with precision, recall,
    f1-score and support for each class, as well as a macro and weighted average
    of each score
    - confusion matrix of model
    '''
    #fit model and time it
    t_start = time.perf_counter()
    classifier.fit(X_train, y_train)
    t_end = time.perf_counter()
    t_train = t_end - t_start

    #score model
    t2_start = time.perf_counter()
    y_pred = classifier.predict(X_test)
    t2_stop = time.perf_counter()
    t_fit = t2_stop - t2_start
    test_scores = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')
    return (t_train, t_fit, test_scores, conf_mat)

def batch_classify_models(classifiers_dict, X_train, y_train, X_test, y_test, verbose=True):
    '''
    Fit and score a number of classifier models for given data.

    Arguments:
    - classifiers_dict: a dictionary containing classifier names keyed to their respective instantiations
    - X_train, y_train: model training data
    - X_test, y_test: model testing data
    - verbose: if verbose, prints training and fit times in seconds for each model during the running process

    Returns:
    - a DataFrame containing for each classifier: length of training time in seconds, length of fit time in seconds, overall accuracy, and precision, recall, f1-score and support for each class, as well as a macro and weighted average of each score
    - a dictionary with classifier names keyed to the confusion matrix of respective model
    - a list of fitted classifier models
    '''
    models_list = []
    models_dict = {}
    confusion_matrices = {}
    for name, classifier in list(classifiers_dict.items()):

        results = fit_score(classifier, X_train, y_train, X_test, y_test)
        t_train, t_fit, test_scores, conf_mat = results

        #add models, model scores, confusion matrices into respective objects
        models_list.append(classifier)
        models_dict[name] = {'train_time': t_train, 'predict_time': t_fit,
                             **test_scores}
        confusion_matrices[name] = conf_mat

        if verbose:
            print(f'Trained {name} in {round(t_train, 3)} seconds.')
            print(f'{name} predicted test data in {round(t_fit, 3)} seconds.')

    scores = pd.DataFrame(models_dict).T
    return (scores, confusion_matrices, models_list)

def batch_classify_data_subsets(classifier, path_dfs, verbose=True):
    '''
    Fit and score a classifier model for a number of data frames containing subsets of features.

    Arguments:
    - classifier: classifier to use
    - path_dfs: a dictionary containing a list of names of various data subsets keyed to a dataframe contained said data subset
    - verbose: if verbose, prints training and fit times in seconds for each model during the running process

    Returns:
    - a DataFrame containing for each data subset: length of training time in seconds, length of fit time in seconds, overall accuracy, and precision, recall, f1-score and support for each class, as well as a macro and weighted average of each score
    - a dictionary with classifier names keyed to the confusion matrix of respective data subset
    - a list of fitted classifier models
    '''
    models_list = []
    model_score_dict = {}
    confusion_matrices = {}
    for (path, df) in path_dfs.items():
        #split data
        X, y = df.iloc[:,1:].sort_index(axis=1), df.iloc[:,0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25, stratify=y)

        results = fit_score(classifier, X_train, y_train, X_test, y_test)
        t_train, t_fit, test_scores, conf_mat = results

        #add models, model scores, confusion matrices into respective objects
        models_list.append(classifier)
        model_score_dict[path] = {'train_time': t_train, 'predict_time': t_fit,
                             **test_scores}
        confusion_matrices[path] = conf_mat

        if verbose:
            print(f'Trained {path} in {round(t_train, 3)} seconds.')
            print(f'{path} predicted test data in {round(t_fit, 3)} seconds.')

    scores = pd.DataFrame(model_score_dict).T
    scores = performance_frame(scores)
    return (scores, confusion_matrices, models_list)

def performance_frame(df):
    '''
    Clean up the scores dataframe.

    The models dataframe from the batch_classify functions is messy. Each entry for a particular class/model combo contains a dict with four entries: precision, recall, f1-score and support. This function expands those dicts and shifts around certain entries to reduce redundancy.

    Arguments:
    - df: scores DataFrame to clean up

    Returns:
    - a DataFrame with columns for each model or data subset and a MultiIndex containing the following levels: 'Metric', 'Type' (aka class name or average type), and 'ClassSize' (aka number of samples in a class)
    '''
    keys = ['A', 'CV', 'CarbonWD', 'F', 'G', 'K', 'LT', 'M', 'OB', 'macro avg', 'weighted avg']
    expand_metrics = []

    #each row for these columns contains a dict with a number of performance metrics
    ##so here I'm expanding these dicts into a new multi-indexed dataframe
    for col in keys:
        if isinstance(df[col][0], str):
            df[col] = df[col].apply(eval)
        expand_metrics.append(df[col].apply(pd.Series).T)
    report = pd.concat(expand_metrics, keys=keys, names=['Type', 'Metric'])

    #support score aka class size is the same for each class across classifiers
    ##so i'm switching it from a row to an index level
    report['ClassSize'] = [x[0] for x in report.index]
    report['ClassSize'] = report['ClassSize'].map({x:report.loc[x, 'support'][0] for x in report.index.levels[0]})
    report_score = report.reset_index().set_index(['Metric', 'Type', 'ClassSize']).sort_index(level='Metric').loc['f1-score':'recall']

    # adding back in the rows from the initial df that contain metrics not separated by class
    add_rows = df[['train_time', 'predict_time', 'accuracy']].T
    add_rows['Type'] = ['overall']*3
    add_rows['ClassSize'] = [report_score.index.levels[2][-1]]*3
    add_rows.index.name = 'Metric'
    add_rows = add_rows.reset_index().set_index(['Metric','Type',
                                                'ClassSize']).sort_index()
    scores = add_rows.append(report_score)
    return scores

def print_confusion_matrices(confusion_matrices, class_names, figsz = (10,7), fontsize=14):
    '''
    Plot a list of confusion matrices as heatmaps, with subplots titled by model or data subset.

    Arguments:
    - confusion_matrices: a dict with name of each model or data subset keyed to respective confusion matrix
    - class_names: list of class names to label heatmap axes

    Returns:
    - a figure containing plotted matrices
    '''
    cmf_mats = sorted(list(confusion_matrices.items()), key=lambda x:np.mean(np.diag(x[1])))
    num_matrices = len(cmf_mats)
    fig, axes = plt.subplots(num_matrices, 1, figsize=(figsz[0], figsz[1]*num_matrices))
    for i, (name, matrix) in enumerate(cmf_mats):
        df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names)
        try:
            if num_matrices > 1:
                heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax=axes[i], cmap='GnBu')
            else:
                heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax=axes, cmap='GnBu')
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)
        heatmap.set_ylabel('True label')
        heatmap.set_xlabel('Predicted label')
        heatmap.set_title(name)
    return fig
