from collections import defaultdict

import numpy as np
from sklearn import svm


def makefeatx(fx, fy, bx, by, img):
    tmpx = np.hstack((fx, bx))
    tmpy = np.hstack((fy, by))
    featx = np.zeros((len(tmpx), 15))

    for i in range(len(tmpx)):
        r = tmpy[i]
        c = tmpx[i]
        featx[i] = np.array([
            img[r][c][0], img[r][c][1], img[r][c][2], img[r][c - 1][0],
            img[r][c - 1][1], img[r][c - 1][2], img[r - 1][c][0],
            img[r - 1][c][1], img[r - 1][c][2], img[r][c + 1][0],
            img[r][c + 1][1], img[r][c + 1][2], img[r + 1][c][0],
            img[r + 1][c][1], img[r + 1][c][2]
        ])

    return featx


def featy(fx, bx):
    n = len(fx) + len(bx)
    y = np.zeros(n)
    for i in range(len(fx)):
        y[i] = 1
    return y


def makeMask(mx, my, smat):
    for i in range(len(mx)):
        r = my[i]
        c = mx[i]
        smat[r][c] = 1
    return smat


def scorestats(preds, scores):
    J1 = []
    J0 = []
    for i in range(len(preds)):
        if preds[i] == 1:
            J1.append(scores[i])
        else:
            J0.append(-scores[i])

    mu1 = np.mean(np.array(J1))
    mu0 = np.mean(np.array(J0))
    std1 = np.std(np.array(J1))
    std0 = np.std(np.array(J0))
    return mu1, mu0, std1, std0


def make_neighbors(nodeids):
    neighdict = defaultdict(list)
    for r in range(nodeids.shape[0]):
        for c in range(nodeids.shape[1]):
            pivot = np.array([r, c])
            # pivot: [left, top, right, bottom]
            val = nodeids[r][c]
            neighdict[val].append(get_neighbor(pivot))
    return neighdict


def getU2(h, w, img, neighmat, flatnodes, lambdaa):
    flatR = img[:, :, 0].flatten()
    flatG = img[:, :, 1].flatten()
    flatB = img[:, :, 2].flatten()
    tU2 = np.zeros((len(flatR)))
    U2dict = defaultdict(dict)
    for u in flatnodes:
        tmpU2 = 0
        tmpr = np.where(u == neighmat[:, 0])[0]
        for v in tmpr:
            npx = int(neighmat[:, 1][v])
            Iv = np.array([flatR[npx], flatG[npx], flatB[npx]])
            Iu = np.array([flatR[u], flatG[u], flatB[u]])
                # individual pixelwise U2
                #pU2 = 1. / (1 + np.linalg.norm(Iu - Iv, 1))
            sigma = 2
            pU2 = np.exp(-(np.linalg.norm(Iu - Iv, 2))**2 / (2 * sigma ^ 2))
            pU2 = (1-lambdaa)*pU2
            U2dict[u][npx] = pU2
            tmpU2 += pU2
        tU2[u] = tmpU2  # sum of all neighbors

    return tU2, U2dict


def get_neighbor(pivot):
    left = pivot + np.array([0, -1])
    top = pivot + np.array([-1, 0])
    right = pivot + np.array([0, 1])
    bottom = pivot + np.array([1, 0])
    neighlist = np.array([left, top, right, bottom])
    return neighlist


def edgeconnected(nodeids):
    h, w = nodeids.shape
    col0 = np.array([])
    col1 = np.array([])
    seq = nodeids.flatten()

    a = seq
    b = a + 1
    delelm = list(a[w - 1::w])
    a = np.delete(a, delelm)
    b = np.delete(b, delelm)

    # for i in b[::w][1:]:
    #     b.remove(b[i])

    # make right
    col0 = np.append(col0, a, axis=0)
    col1 = np.append(col1, b, axis=0)

    # make left
    col0 = np.append(col0, b, axis=0)
    col1 = np.append(col1, a, axis=0)

    # make down
    a = seq
    a = a[:(len(a) - w)]
    b = a + w
    col0 = np.append(col0, a, axis=0)
    col1 = np.append(col1, b, axis=0)

    # make up
    col0 = np.append(col0, b, axis=0)
    col1 = np.append(col1, a, axis=0)

    neighmat = np.array([col0, col1]).T

    return neighmat


def getU1(img, model, neighmat, flatnodes, mu1, mu0, std1, std0):
    U1 = np.zeros((len(flatnodes), 2))
    preds = []

    # Make feature x
    flatR = img[:, :, 0].flatten()
    flatG = img[:, :, 1].flatten()
    flatB = img[:, :, 2].flatten()
    for u in flatnodes:
        Iu = [flatR[u], flatG[u], flatB[u]]
        featx = [flatR[u], flatG[u], flatB[u]]
        tmpr = np.where(u == neighmat[:, 0])[0]
        for v in tmpr:
            npx = int(neighmat[:, 1][v])
            Iv = [flatR[npx], flatG[npx], flatB[npx]]
            featx.extend(Iv)

        if len(featx) < 15:
            ls = (15 - len(featx)) / 3
            for i in range(ls):
                featx.extend(Iu)

        # Predict on featx
        featx = np.array(featx).reshape(-1, 15)
        pred = model.predict(featx)
        score = model.decision_function(featx)
        preds.append(pred)


        if pred == 1:
            p = 1 / (1 + np.exp(-4 * score / mu1))
            assert (p <=1), "Probability not less than 1"
            U1[u, 0] = p  # bgcost
            U1[u, 1] = 1 - p  # fgcost

        if pred == 0:
            score = -score
            p = 1 / (1 + np.exp((-4 * score) / mu0))
            assert (p <= 1), "Probability not less than 1"
            U1[u, 1] = p  # fgcost
            U1[u, 0] = 1 - p  # bgcost


    return U1, preds
