# %%
from collections import defaultdict
from sklearn import linear_model
import numpy
import math

def cal_BER(y_true, pred):
    TP = sum(numpy.logical_and(y_true, pred))
    TN = sum(numpy.logical_and(numpy.logical_not(y_true), numpy.logical_not(pred)))
    FP = sum(numpy.logical_and(numpy.logical_not(y_true), pred))
    FN = sum(numpy.logical_and(y_true, numpy.logical_not(pred)))

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)

    BER = 1 - 0.5*(TPR + TNR)
    return BER


def feat(d, catID, maxLength, includeCat = True, includeReview = True, includeLength = True):
    feat = []
    if includeCat:
        one_hot = [0] * len(catID)
        if d['beer/style'] in catID:
            one_hot[catID[d['beer/style']]] = 1
        feat.extend(one_hot)

    if includeReview:
        reviews = [d['review/overall'], d['review/aroma'], d['review/taste'], d['review/appearance'], d['review/palate']]
        feat.extend(reviews)
    

    if includeLength:
        lenNorm = [len(d['review/text'])/maxLength]
        feat.extend(lenNorm)
    
    return feat + [1]


def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced', max_iter=5000)

    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    Xtrain = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    Xvalid = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    Xtest = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]
    
    # (1) Fit the model on the training set
    mod.fit(Xtrain, yTrain)
    
    # (2) Compute validation BER
    predValid = mod.predict(Xvalid)
    vBER = cal_BER(yValid, predValid)

    # (3) Compute test BER
    predTest = mod.predict(Xtest)
    tBER = cal_BER(yTest, predTest)

    return mod, vBER, tBER


def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)
    return mod, validBER, testBER


def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)
    return mod, validBER, testBER


def Q3(catID, dataTrain, dataValid, dataTest):
    # Your solution here...
    bestMod = None
    vBest, tBest = float('inf'), float('inf')
    for c in [0.001, 0.01, 0.1, 1, 10]:
        mod, vBER, tBER = pipeline(c, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True)
        if vBER < vBest:
            bestMod = mod
            vBest = vBER
            tBest = tBER

    return bestMod, vBest, tBest


def Q4(C, catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    return testBER_noCat, testBER_noReview, testBER_noLength


def Jaccard(s1, s2):
    # Implement
    nume = len(s1.intersection(s2))
    deno = len(s1.union(s2))
    if deno == 0: return 0
    return nume/deno

def mostSimilar(i, N, usersPerItem):
    # Implement...
    similarities = []
    U_i = usersPerItem[i]
    for j in usersPerItem:
        if j != i:
            U_j = usersPerItem[j]
            Jsim = Jaccard(U_i, U_j)
            similarities.append((Jsim, j))
    similarities.sort(reverse=True)
    return similarities[:N]

def MSE(y, ypred):
    diff = [(ypred[i] - y[i])**2 for i in range(len(y))]
    sum_diff = sum(diff)
    return sum_diff/len(y)

def getMeanRating(dataTrain):
    # Implement...
    lenData = len(dataTrain)
    listData = sum([d['star_rating'] for d in dataTrain])

    return listData/lenData

def getUserAverages(itemsPerUser, ratingDict):
    # Implement (should return a dictionary mapping users to their averages)
    userAverages = defaultdict(float) # user: avg. rating
    for user, items in itemsPerUser.items():
        ratingList = [ratingDict[(user,item)] for item in items]
        lenList = len(ratingList)
        sumList = sum(ratingList)
        userAverages[user] = sumList/lenList

    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    # Implement...
    itemAverages = defaultdict(float)
    for item, users in usersPerItem.items():
        ratingList = [ratingDict[(user,item)] for user in users]
        lenList = len(ratingList)
        sumList = sum(ratingList)
        itemAverages[item] = sumList/lenList
        
    return itemAverages



def predictRating(user, item, ratingMean, reviewsPerUser,
                  usersPerItem, itemsPerUser, userAverages, itemAverages):
    if item not in itemAverages:
        return ratingMean

    R_i = itemAverages[item]
    nume = 0
    deno = 0

    for d in reviewsPerUser.get(user, []):
        item_j = d['product_id']

        if item_j == item:
            continue

        R_uj = d['star_rating']
        R_j = itemAverages[item_j]
        sim_ij = Jaccard(usersPerItem[item], usersPerItem[item_j])

        nume += (R_uj - R_j) * sim_ij
        deno += sim_ij

    return R_i + nume / deno if deno != 0 else R_i



def predictRatingQ7(user, item, ratingMean, reviewsPerUser,
                    usersPerItem, itemsPerUser,
                    userAverages, itemAverages):
    user_count = len(itemsPerUser.get(user, set()))
    item_count = len(usersPerItem.get(item, set()))

    user_bias = 0
    if user_count:
        user_bias = (
            user_count / (user_count + 10)
            * (userAverages[user] - ratingMean)
        )

    item_bias = 0
    if item_count:
        item_bias = (
            item_count / (item_count + 10)
            * (itemAverages[item] - ratingMean)
        )

    prediction = ratingMean + user_bias + item_bias
    return float(numpy.clip(prediction, 1, 5))