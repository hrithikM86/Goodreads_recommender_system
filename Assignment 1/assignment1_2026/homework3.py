# Converted from baselines.ipynb in hrithikM86/Goodreads_recommender_system.
# Run from the folder containing the assignment training and test CSV files.
# Dependencies: numpy scipy implicit scikit-surprise tqdm

import gzip
import random
import scipy
import numpy
from collections import defaultdict
from implicit import bpr
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from tqdm import tqdm as tqdm

dataDir = "/Users/hrithik/Desktop/UC San Diego/Courses/UCSD CSE 258/Assignment 1/assignment1_2026"

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
            est = model.predict(u,i).est
            b.write(l.strip() + ',' + str(est) + '\n')
        except Exception as e:
            print(e)
    a.close()
    b.close()

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
    
    model = bpr.BayesianPersonalizedRanking(factors = 32)
    
    model.fit(Xui_csr)
    return model, userIDs, itemIDs

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

# Task 1: latent factor model

trainset,m1 = latentFactorModel(dataDir + '/' + "tasks_1_2_train.csv")

testPredictionsLFM(trainset, m1, "task_1_test.csv", "task_1_predictions.csv")

# Task 2: BPR

m2,userIDs,itemIDs = BPRModel(dataDir + '/' + "tasks_1_2_train.csv")

testPredictionsBPR(m2, userIDs, itemIDs, "task_2_test.csv", "task_2_predictions.csv")

# Task 3: latent factor model, just ignore the timestamp

trainset,m3 = latentFactorModel(dataDir + '/' + "tasks_3_4_train.csv")

testPredictionsLFM(trainset, m3, "task_3_test.csv", "task_3_predictions.csv") # Don't use the timestamp

# Task 4: BPR, just ignore the timestamp

m4,userIDs,itemIDs = BPRModel(dataDir + '/' + "tasks_3_4_train.csv")

testPredictionsBPR(m4, userIDs, itemIDs, "task_4_test.csv", "task_4_predictions.csv")

# Task 5: BPR, ignore timestamp/ordering

m5,userIDs,itemIDs = BPRModel(dataDir + '/' + "task_5_train.csv")

testPredictionsBPR(m5, userIDs, itemIDs, "task_5_test.csv", "task_5_predictions.csv")

# Task 6: compute median delta

a = open("task_6_train.csv")
a.readline()
userDeltas = defaultdict(list)
uLast = {} # Time of user's most recent interaction
for l in a.readlines():
    u,i,t,r = l.strip().split(',')
    t = float(t)
    if u in uLast:
        userDeltas[u].append(t - uLast[u])
    uLast[u] = t
a.close()

medians = {}
for u in userDeltas:
    medians[u] = numpy.mean(userDeltas[u])

a = open("task_6_test.csv", 'r')
b = open("task_6_predictions.csv", 'w')
b.write(a.readline()) # header
for l in a.readlines():
    u,i = l.strip().split(',')
    b.write(u + ',' + i + ',' + str(medians[u]) + '\n')

b.close()

# Baseline performance:

# public =  {"task1": 0.7429, "task2":0.8716, "task3":0.9820, "task4":0.8036, "task5":0.8189, "task6":13563799}
# private = {"task1": 0.7738, "task2":0.8736, "task3":0.9582, "task4":0.8007, "task5":0.8195, "task6":13481806}


