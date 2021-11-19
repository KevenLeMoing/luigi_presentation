import luigi
import csv
import time
import pickle
import requests
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class DownloadDatasetTask(luigi.Task):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    output = 'exercise_2/db/download_data.csv'

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget(self.output)]

    def run(self):
        response = requests.get(self.url)
        with open('exercise_2/db/download_data.csv', 'w') as f:
            writer = csv.writer(f)
            for line in response.iter_lines():
                writer.writerow(line.decode('utf-8').split(';'))


class SplitDatasetTask(luigi.Task):
    input = luigi.Parameter(default='exercise_2/db/download_data.csv')
    x_train = luigi.Parameter(default='exercise_2/db/x_train.csv')
    x_test = luigi.Parameter(default='exercise_2/db/x_test.csv')
    y_train = luigi.Parameter(default='exercise_2/db/y_train.csv')
    y_test = luigi.Parameter(default='exercise_2/db/y_test.csv')

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget(self.x_train),
                luigi.LocalTarget(self.x_test),
                luigi.LocalTarget(self.y_train),
                luigi.LocalTarget(self.y_test)]

    def run(self):
        df = pd.read_csv(self.input)
        features = df.drop('quality', axis=1)
        label = df['quality']
        x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
        x_train.to_csv(self.x_train)
        y_train.to_csv(self.y_train)
        x_test.to_csv(self.x_test)
        y_test.to_csv(self.y_test)


class TrainModelTask(luigi.Task):
    x_train = luigi.Parameter(default='exercise_2/db/x_train.csv')
    y_train = luigi.Parameter(default='exercise_2/db/y_train.csv')
    model_train = luigi.Parameter(default='exercise_2/db/model_train.pkl')

    def requires(self):
        return [SplitDatasetTask()]

    def output(self):
        return [luigi.LocalTarget(self.model_train)]

    def run(self):
        x_train = pd.read_csv(self.x_train)
        y_train = pd.read_csv(self.y_train)
        regr = linear_model.LinearRegression()
        regr = regr.fit(x_train, y_train)
        pickle.dump(regr, open(self.model_train, 'wb'))


class EvaluateModelTask(luigi.Task):
    x_test = luigi.Parameter(default='exercise_2/db/x_test.csv')
    y_test = luigi.Parameter(default='exercise_2/db/y_test.csv')
    train_path = luigi.Parameter(default='exercise_2/db/model_train.pkl')
    serving_path = luigi.Parameter(default='exercise_2/db/serving/model_served.pkl')

    def requires(self):
        return [SplitDatasetTask(), TrainModelTask()]

    def output(self):
        return [luigi.LocalTarget(self.serving_path)]

    def run(self):
        model_train = pickle.load(open(self.train_path, 'rb'))
        x_test = pd.read_csv(self.x_test)
        y_test = pd.read_csv(self.y_test)
        model_score = model_train.score(x_test, y_test)
        if model_score > 0.5:
            pickle.dump(model_train, open(self.serving_path, 'wb'))
        else:
            print('-----------------------------------------')
            print('TOO BAD MODEL')
            print('-----------------------------------------')
