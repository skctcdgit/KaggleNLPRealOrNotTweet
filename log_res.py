import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    # read train.csv
    df = pd.read_csv("G:/Kaggle/Real or Not NLP with Disaster Tweets/train.csv")

    validation_df = pd.read_csv(
        "G:/Kaggle/Real or Not NLP with Disaster Tweets/test.csv"
    )

    # create kfold column
    df["kfold"] = -1

    # randomize dataframe rows
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch target label
    y = df.target.values

    # print(y)
    # print(len(y))

    # kfold initiation and fill kfold column
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    # creating test and train dataframes
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # countvectorizer initialization
        cv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

        # print(cv)

        # fit countvectorizer to the real/fake tweets (tweet text)
        cv.fit(train_df.text)

        # transform into sparse term-document matrix
        xtrain = cv.transform(train_df.text)
        xtest = cv.transform(test_df.text)
        xvalidate = cv.transform(validation_df.text)

        # print(xtrain)
        # print("_"*50)
        # print(xtest)

        # Logistic Regression Model
        logistic_model = linear_model.LogisticRegression(solver="liblinear")

        # fit logistic model
        logistic_model.fit(xtrain, train_df.target)

        # predict on test data
        pred = logistic_model.predict(xtest)

        # measure accuracy (?)
        accuracy = metrics.accuracy_score(test_df.target, pred)

        print(f"Fold: {fold_}")
        print(f"Accuracy: {accuracy}")
        print("")

    pred_validation_df = logistic_model.predict(xvalidate)

    # print(len(validation_df["id"]))
    # print(len(pred_validation_df))

    submission = pd.DataFrame(
        {"id": list(validation_df["id"]), "target": list(pred_validation_df)}
    )

submission.to_csv(
    "G:\Kaggle\Real or Not NLP with Disaster Tweets\submission.csv", index=False
)
