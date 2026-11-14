# Converted from baselines.ipynb in hrithikM86/Goodreads_recommender_system.
# Run from the directory containing the assignment CSV files.

# %%
# ! pip install scipy

# %%
# ! pip install implicit

# %%
# ! pip install surprise

# %%
# ! pip install pandas

# %%
# ! pip install scikit-learn

# %%
import gzip
import random
import scipy
import numpy
import csv
import random
import pandas as pd
from collections import defaultdict
from implicit import bpr
from surprise import SVD, Reader, Dataset, SVDpp
from surprise.model_selection import train_test_split
from tqdm import tqdm as tqdm
from datetime import datetime, timezone
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression


# %%
import os
print(os.getcwd())

# %%
dataDir = "/Users/hrithik/Desktop/UC San Diego/Courses/UCSD CSE 258/Assignment 1/assignment1_2026"

# %%
def latentFactorModel(trainFile):
    with open(trainFile, 'r') as a, open("temp.csv", 'w') as b:
        a.readline()  
        for line in a:
            b.write(line)

    reader = Reader(line_format='user item timestamp rating', sep=',')
    data = Dataset.load_from_file("temp.csv", reader=reader)

    model = SVD(
        n_factors=16,
        n_epochs=50,
        reg_all=0.1,
        random_state=42
    )

    trainset, testset = train_test_split(
        data, test_size=.25, random_state=42
    )
    model.fit(trainset)

    predictions = model.test(testset)
    mse = sum((p.r_ui - p.est) ** 2 for p in predictions) / len(predictions)
    print("Validation MSE =", mse)

    full_trainset = data.build_full_trainset()
    model.fit(full_trainset)

    return full_trainset, model

# %%
def testPredictionsLFM(trainset, model, testIn, testOut):
    a = open(testIn, 'r')
    b = open(testOut, 'w')
    header = a.readline()
    b.write(header)
    for l in a.readlines():
        fields = l.split(',')
        u,i = fields[0].strip(),fields[1].strip()
        try:
            #print((u,i))
            est = model.predict(u,i).est #Ratings
            b.write(l.strip() + ',' + str(est) + '\n')
        except Exception as e:
            print(e)
    a.close()
    b.close()

# %%
def BPRModel(trainFile):
    a = open(trainFile, 'r')
    a.readline() # header
    
    data = []
    userIDs,itemIDs = {},{}
    
    for l in a.readlines():
        fields = l.split(',')
        u,i = fields[0].strip(),fields[1].strip()
        data.append((u,i))
        if not u in userIDs: userIDs[u] = len(userIDs)
        if not i in itemIDs: itemIDs[i] = len(itemIDs)
        
    nUsers,nItems = len(userIDs),len(itemIDs)

    Xui = scipy.sparse.lil_matrix((nUsers, nItems))
    for u,i in data:
        Xui[userIDs[u],itemIDs[i]] = 1

    Xui_csr = scipy.sparse.csr_matrix(Xui)
    
    model = bpr.BayesianumpyersonalizedRanking(factors = 32)
    
    model.fit(Xui_csr)
    return model, userIDs, itemIDs

# %%
def testPredictionsBPR(model, userIDs, itemIDs, testIn, testOut):
    a = open(testIn, 'r')
    b = open(testOut, 'w')
    header = a.readline()
    b.write(header)
    randomGuesses = 0
    for l in a.readlines():
        fields = l.split(',')
        # User is first field, items are second last and last fields
        u,i1,i2 = fields[0].strip(),fields[-2].strip(),fields[-1].strip()
        try:
            u_ = userIDs[u]
            i1_ = itemIDs[i1]
            i2_ = itemIDs[i2]
        except Exception as e:
            #print("Missing user or item, guessing randomly")
            b.write(l.strip() + ',' + random.choice([i1,i2]) + '\n')
            randomGuesses += 1
        try:
            uf = model.user_factors[u_]
            i1f = model.item_factors[i1_]
            i2f = model.item_factors[i2_]
            est = i1
            if uf.dot(i2f) > uf.dot(i1f):
                est = i2
            b.write(l.strip() + ',' + est + '\n')
        except Exception as e:
            print(e)
    a.close()
    b.close()
    print("Guessed randomly " + str(randomGuesses) + " times")

# %%
def monthKey(timestamp):
    dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return dt.year, dt.month


class TemporalSVD:
    def __init__(self, month_reg=100):
        self.month_reg = month_reg
        self.model = SVD(
            n_factors=16,
            n_epochs=50,
            reg_all=0.1,
            random_state=42
        )

    def fit(self, trainset, timestamps):
        self.model.fit(trainset)

        residual_sums = defaultdict(float)
        counts = defaultdict(int)

        for (u, i, rating), timestamp in zip(
            trainset.all_ratings(), timestamps
        ):
            user = trainset.to_raw_uid(u)
            item = trainset.to_raw_iid(i)

            baseline = self.model.predict(user, item).est
            month = monthKey(timestamp)

            residual_sums[month] += rating - baseline
            counts[month] += 1

        self.month_bias = {
            month: total / (counts[month] + self.month_reg)
            for month, total in residual_sums.items()
        }

        self.rating_scale = trainset.rating_scale
        return self

    def predict(self, user, item, timestamp):
        baseline = self.model.predict(user, item).est
        correction = self.month_bias.get(monthKey(timestamp), 0.0)

        low, high = self.rating_scale
        estimate = max(low, min(high, baseline + correction))

        return SimpleNamespace(est=estimate)

# %%
def temporalFactorModel(trainFile):
    with open(trainFile, newline="") as f:
        rows = list(csv.reader(f))[1:]

    rows = [
        (u, i, float(t), float(r))
        for u, i, t, r in rows
    ]

    random.Random(42).shuffle(rows)
    split = int(len(rows) * 0.75)
    training, validation = rows[:split], rows[split:]

    def fit_model(records):
        frame = pd.DataFrame(
            records, columns=["user", "item", "timestamp", "rating"]
        )

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            frame[["user", "item", "rating"]], reader
        )
        trainset = data.build_full_trainset()

        # Match timestamps to Surprise's internal iteration order.
        times_by_pair = defaultdict(list)
        for u, i, t, r in records:
            times_by_pair[(u, i, r)].append(t)

        aligned_times = []
        for u, i, r in trainset.all_ratings():
            key = (
                trainset.to_raw_uid(u),
                trainset.to_raw_iid(i),
                r
            )
            aligned_times.append(times_by_pair[key].pop())

        return TemporalSVD(month_reg=100).fit(
            trainset, aligned_times
        )

    model = fit_model(training)

    baseline_mse = sum(
        (r - model.model.predict(u, i).est) ** 2
        for u, i, t, r in validation
    ) / len(validation)

    temporal_mse = sum(
        (r - model.predict(u, i, t).est) ** 2
        for u, i, t, r in validation
    ) / len(validation)

    print("SVD validation MSE:", baseline_mse)
    print("Temporal validation MSE:", temporal_mse)

    # Final submission model
    return fit_model(rows)

# %%
def testPredictionsTemporal(model, testIn, testOut):
    with open(testIn) as a, open(testOut, "w") as b:
        b.write(a.readline())

        for line in a:
            fields = line.strip().split(",")
            user, item, timestamp = fields[:3]

            prediction = model.predict(user, item, timestamp).est
            b.write(line.strip() + "," + str(prediction) + "\n")

# %%


# %%
# Task 1: latent factor model

# %%
trainset,m1 = latentFactorModel(dataDir + '/' + "tasks_1_2_train.csv")

# %%
testPredictionsLFM(trainset, m1, "task_1_test.csv", "task_1_predictions.csv")

# %%
# Task 2: BPR

# %%
m2,userIDs,itemIDs = BPRModel(dataDir + '/' + "tasks_1_2_train.csv")

# %%
testPredictionsBPR(m2, userIDs, itemIDs, "task_2_test.csv", "task_2_predictions.csv")

# %%
# Task 3: latent factor model, just ignore the timestamp

# %%
# trainset,m3 = latentFactorModel(dataDir + '/' + "tasks_3_4_train.csv")

# %%
# testPredictionsLFM(trainset, m3, "task_3_test.csv", "task_3_predictions.csv") # Don't use the timestamp

# %%
m3 = temporalFactorModel(dataDir + "/tasks_3_4_train.csv")

# %%
testPredictionsTemporal(
    m3,
    dataDir + "/task_3_test.csv",
    dataDir + "/task_3_predictions.csv"
)

# %%
# Task 4: BPR, just ignore the timestamp

# %%
m4,userIDs,itemIDs = BPRModel(dataDir + '/' + "tasks_3_4_train.csv")

# %%
testPredictionsBPR(m4, userIDs, itemIDs, "task_4_test.csv", "task_4_predictions.csv")

# %%
# Task 5: BPR, ignore timestamp/ordering

# %%
m5,userIDs,itemIDs = BPRModel(dataDir + '/' + "task_5_train.csv")

# %%
testPredictionsBPR(m5, userIDs, itemIDs, "task_5_test.csv", "task_5_predictions.csv")

# %%
# Task 6: compute median delta

# %%
# # EMA_t = alpha * x_t + (1-alpha) * EMA_{t-1}
# def EMA(values, alpha = 0.1):
#     ema = values[0]
#     for value in values[1:]:
#         ema = alpha * value + (1-alpha) * ema

#     return ema

# %%
# a = open("task_6_train.csv")
# a.readline()
# userDeltas = defaultdict(list)
# uLast = {} # Time of user's most recent interaction
# for l in a.readlines():
#     u,i,t,r = l.strip().split(',')
#     t = float(t)
#     if u in uLast:
#         userDeltas[u].append(t - uLast[u])
#     uLast[u] = t
# a.close()

# ema = {}
# for u in userDeltas:
#     ema[u] = EMA(userDeltas[u])

# a = open("task_6_test.csv", 'r')
# b = open("task_6_predictions.csv", 'w')
# b.write(a.readline()) # header
# for l in a.readlines():
#     u,i = l.strip().split(',')
#     b.write(u + ',' + i + ',' + str(ema[u]) + '\n')

# b.close()

# %%
userDeltas = defaultdict(list)
uLast = {}

with open("task_6_train.csv") as a:
    a.readline()
    for line in a:
        u, i, t, r = line.strip().split(',')
        t = float(t)
        if u in uLast:
            userDeltas[u].append((t - uLast[u]) / 86400)
        uLast[u] = t


def features(gaps):
    return [gaps[-1], numpy.median(gaps), numpy.mean(gaps)]


X, y = [], []
for gaps in userDeltas.values():
    for k in range(1, len(gaps)):
        X.append(features(gaps[:k]))
        y.append(gaps[k])

model = LinearRegression()
model.fit(X, y)

estimates = {
    u: max(0.0, model.predict([features(gaps)])[0]) * 86400
    for u, gaps in userDeltas.items()
}

with open("task_6_test.csv") as a, \
        open("task_6_predictions.csv", "w") as b:
    b.write(a.readline())
    for line in a:
        u, i = line.strip().split(',')
        b.write(f"{u},{i},{estimates[u]}\n")

# %%
# Baseline performance:

# public =  {"task1": 0.7429, "task2":0.8716, "task3":0.9820, "task4":0.8036, "task5":0.8189, "task6":13563799}
# private = {"task1": 0.7738, "task2":0.8736, "task3":0.9582, "task4":0.8007, "task5":0.8195, "task6":13481806}

# %%
print('Done')

# %%


