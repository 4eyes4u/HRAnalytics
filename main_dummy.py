import os
import time

import numpy as np
import pandas as pd
import category_encoders as ce

from xverse.ensemble import VotingSelector
from argparse import Namespace
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real
from skopt.callbacks import VerboseCallback

INPUT_PATH = "data/input/"
OUTPUT_PATH = "data/output"
INDEX_COL = "EmployeeID"
TARGET_COL = "Attrition"

params = {
    "random_state": 1337,
    "C_lower": 1e-3,  # lower bound for C search
    "C_upper": 1e3,  # upper bound for C search
    "train_holdout_size": 0.2,  # train/val_test size ratio
    "val_test_size": 0.5,  # val/test size ratio
    "n_jobs": 12,  # number of jobs to run in parallel for Bayes search
    "n_iters": 100,  # number of iterations of Bayes search
    "base_estimator": "RF"  # random forest esimator for Bayes search
}


def create_dummies(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [col for col in X.select_dtypes("object").columns if
                        col not in [TARGET_COL, INDEX_COL]]

    dummy_df = pd.get_dummies(X,
                              columns=categorical_cols,
                              prefix=categorical_cols,
                              drop_first=True)

    return dummy_df


def report_performance(optimizer: BayesSearchCV,
                       X: pd.DataFrame,
                       y: pd.DataFrame,
                       callbacks=None
                       ) -> dict:
    r"""Verbose for optimizer. Will call callbacks in each iteration."""

    optimizer.fit(X, y, callback=callbacks)

    return optimizer.best_params_


if __name__ == "__main__":
    params = Namespace(**params)

    # directory for saving results
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # merging all available data sets
    data = pd.read_csv(os.path.join(INPUT_PATH, "general_data.csv"))
    mgr_surveys = pd.read_csv(os.path.join(INPUT_PATH,
                                           "manager_survey_data.csv"))
    emp_surveys = pd.read_csv(os.path.join(INPUT_PATH,
                                           "employee_survey_data.csv"))

    surveys = pd.merge(mgr_surveys, emp_surveys, on=INDEX_COL)
    data = pd.merge(data, surveys, on=INDEX_COL)

    # mapping target class in binary value and removing from training data
    y = data["Attrition"].replace({"Yes": 1, "No": 0})
    X = data.drop(labels="Attrition", axis=1)
    X = create_dummies(X)

    # splitting data in train/val/test sets
    X_train, X_holdout, y_train, y_holdout = \
        train_test_split(X, y,
                         test_size=params.train_holdout_size,
                         random_state=params.random_state,
                         stratify=y)
    X_val, X_test, y_val, y_test = \
        train_test_split(X_holdout, y_holdout,
                         test_size=params.val_test_size,
                         random_state=params.random_state,
                         stratify=y_holdout)

    # discarding irrelevant features
    selectors = ["WOE", "RF", "ETC", "CS", "L_ONE"]
    feature_selector = VotingSelector(minimum_votes=3,
                                      selection_techniques=selectors)
    feature_selector.fit(X_train, y_train)
    X_train = feature_selector.transform(X_train)
    X_val = feature_selector.transform(X_val)
    X_test = feature_selector.transform(X_test)

    # scaling to N(0, 1)
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_val = standard_scaler.transform(X_val)
    X_test = standard_scaler.transform(X_test)

    # training LR with class balancing (positive class is less present)
    model = LogisticRegression(class_weight="balanced",
                               C=1.0,
                               max_iter=100)
    model.fit(X_train, y_train)

    # fetching predictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # will be appplied on training set only
    cv_generator = StratifiedKFold(n_splits=5,
                                   shuffle=True,
                                   random_state=params.random_state)

    # exhausting only C with log-uniform distirbution
    distribution = dict(C=Real(params.C_lower,
                               params.C_upper,
                               "log-uniform"))

    # searching for parameter with underlying distribution
    optimizer = BayesSearchCV(model,
                              distribution,
                              cv=cv_generator,
                              scoring="roc_auc_ovr_weighted",
                              n_iter=params.n_iters,
                              n_jobs=params.n_jobs,
                              return_train_score=False,
                              optimizer_kwargs={
                                  "base_estimator": params.base_estimator
                                  },
                              refit=True)

    # optimizing regularization parameter (C) and saving results
    time_path = os.path.join(OUTPUT_PATH, time.strftime("%Y%m%d-%H%M%S"))
    log_path, fig_path = time_path + ".txt", time_path + ".svg"
    with open(log_path, "w") as f:
        best_params = \
            report_performance(optimizer,
                               X_train, y_train,
                               callbacks=[VerboseCallback(params.n_iters)])

        cv_results = pd.DataFrame(optimizer.cv_results_)
        cv_results.sort_values(by=["param_C", "mean_test_score"], inplace=True)
        cv_results.plot(x="param_C",
                        y="mean_test_score",
                        logx=True,
                        sort_columns=True
                        )
        plt.savefig(fig_path, format="svg")

        # performance is checked on validation set in parameter tuning
        print(__file__, file=f)
        print("Parameters for training:", params, file=f)
        print("Logistic regression coeficients:", model.coef_, file=f)
        print("Best_params:", best_params, file=f)

        print("Validation test stats:", file=f)
        print(confusion_matrix(y_val, y_val_pred), file=f)
        print(classification_report(y_val, y_val_pred), file=f)
        print("Test set stats:", file=f)
        print(confusion_matrix(y_test, y_test_pred), file=f)
        print(classification_report(y_test, y_test_pred), file=f)
