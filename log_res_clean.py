"""
@author: SuhridKrishna
"""

import pandas as pd
import re
import string

import nltk
from nltk.tokenize import word_tokenize

# nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    # read train.csv
    df = pd.read_csv("G:/Kaggle/Real or Not NLP with Disaster Tweets/train.csv")

    validation_df = pd.read_csv(
        "G:/Kaggle/Real or Not NLP with Disaster Tweets/test.csv"
    )

    def text_cleaner(df):
        # split strings by spaces
        # str = str.split()
        # remove unnecessary spacings
        # str = " ".join(str)
        # remove punctuations
        # str = re.sub(f"[{re.escape(string.punctuation)}]", "", str)

        # code inspired from https://towardsdatascience.com/text-cleaning-methods-for-natural-language-processing-f2fc1796e8c7
        # removing noise from the text such as punctuations and spaces
        df["text"] = df["text"].str.lower()
        df["text"] = df["text"].apply(
            lambda item: re.sub(
                r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", item
            )
        )
        # remove numbers
        df["text"] = df["text"].apply(lambda item: re.sub(r"\d+", "", item))

        return df

    # code inspired from https://towardsdatascience.com/text-cleaning-methods-for-natural-language-processing-f2fc1796e8c7
    # Lemmatization
    def text_lemmatizer(s):
        lemma = [WordNetLemmatizer().lemmatize(i) for i in s]
        return lemma

    # Stemming
    def text_stemmer(s):
        stemmer = SnowballStemmer("english")
        stem_text = [stemmer.stem(i) for i in s]
        return stem_text

    # apply clean text function
    # df.loc[:, "text"] = df.text.apply(text_cleaner)

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
        train_df = text_cleaner(train_df)
        test_df = text_cleaner(test_df)
        # tfidf initialization

        tfidf = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

        # print(cv)

        # fit tfidf to the real/fake tweets (tweet text)
        tfidf.fit(train_df.text)

        # transform into sparse term-document matrix
        xtrain = tfidf.transform(train_df.text)
        xtest = tfidf.transform(test_df.text)
        xvalidate = tfidf.transform(validation_df.text)
        print(xtrain)

        """
        xtrain = pd.DataFrame(train_df.iloc[:,:-1])
        xtest = pd.DataFrame(test_df.iloc[:,:-1])
        xvalidate = pd.DataFrame(validation_df.iloc[:,:-1])

        print(train_df.head(10))
        print("__" * 25)
        train_df["text_lemma"] = train_df["text"].apply(lambda t: text_lemmatizer(t))
        train_df["text_stem"] = train_df["text"].apply(lambda t: text_stemmer(t))
        print(train_df.head(10))
        # print(xtrain)
        # print("_"*50)
        # print(xtest)
        """
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


"""
submission.to_csv(
    "G:\Kaggle\Real or Not NLP with Disaster Tweets\mlframework\src\submissions\submission_logres_clean.csv",
    index=False,
)
"""
