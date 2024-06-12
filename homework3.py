# Created and formatted by Zachary Wittmann
# Version 1.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plotData(names):
    # Plots every item after the first element against the last element
    for name in names:
        plt.figure()
        plt.scatter(df[name], df[df.columns[len(df.columns)-1]])
        plt.xlabel(name)
        plt.ylabel(df.columns[len(df.columns)-1])


def predictionInfo(tdf, model, printValues=False):
    testData = tdf.values[:, range(1, len(tdf.columns))]

    tdfIDList = tdf[tdf.columns[0]].tolist()
    testUnrankedDict = dict()

    print('\nPrediction:\n')
    for position, item in enumerate(model.predict(testData)):
        testUnrankedDict[tdfIDList[position]] = item

    testRankedDict = sorted(
        testUnrankedDict, key=testUnrankedDict.get, reverse=True)

    for position, val in enumerate(testRankedDict):
        values = f". {testUnrankedDict[val]}" if printValues else ""
        print(f'{position + 1}: {val}{values}')


if __name__ == "__main__":

    dataCSV = 'GrowthData2.csv'
    unknownCSV = 'Growth-unknown.csv'
    # dataCSV = 'JetData.csv'
    # unknownCSV = 'JetData-unknown.csv'

    df = pd.read_csv(dataCSV)
    tdf = pd.read_csv(unknownCSV)

    # Ignore first element (ID) and last element (what we want)
    names = df.columns[range(1, len(df.columns)-1)]

    X = df.values[:, range(1, len(df.columns)-1)]
    y = df.values[:, len(df.columns)-1]

    model = LinearRegression().fit(X, y)

    plotData(names)

    print(f'Utilizing Dataset: {dataCSV}')
    print(f'Utilizing Unknown: {unknownCSV}')

    print('\nCoefficient:')
    for position, item in enumerate(model.coef_.flatten()):
        print(f'{names[position]}: {round(item, 3)}')

    print(f'\nBeta naught:\n{model.intercept_}')

    print(
        f'\nMost Influential Variable:\n{names[np.argmax(abs(model.coef_.flatten()))]}')

    predictionInfo(tdf, model, printValues=True)

    print(f'\nScore:\n{model.score(X, y)}')
