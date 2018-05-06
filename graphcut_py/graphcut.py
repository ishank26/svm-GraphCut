import cv2
import maxflow
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn import svm

from utils import getpixelpts as gp
from utils import modelutils as mut


filename = 'church'

filepath = './results/' + filename + '.jpg'

oimg = cv2.imread(filepath)
h, w = oimg.shape[:2]  # rowxcol format
oimg = cv2.resize(oimg, (int(0.6 * w), int(0.6 * h)), interpolation=cv2.INTER_CUBIC)

ksize = (3, 3)
img = cv2.blur(oimg, ksize)
h, w = img.shape[:2]
N = h * w
# x- column, y - row
fx, fy = gp.getpixels(img, 'fg')
bx, by = gp.getpixels(img, 'bg')

img = cv2.blur(oimg, ksize)
#fgmask = mut.makeMask(fx, fy, csr_matrix((h, w), dtype=np.int8).toarray())
#bgmask = mut.makeMask(bx, by, csr_matrix((h, w), dtype=np.int8).toarray())

X = mut.makefeatx(fx, fy, bx, by, img)
Y = mut.featy(fx, bx)
sigmasq = 1000.
C = 130.
model = svm.SVC(kernel='rbf', gamma=1 / (2 * sigmasq), C=C)
model = model.fit(X, Y)
print "\nTraining accuracy: ", model.score(X, Y)
preds = model.predict(X)
scores = model.decision_function(X)
#probs = model.predict_proba(X)

mu1, mu0, std1, std0 = mut.scorestats(preds, scores)
imgvec = img.reshape(-1, 1)

print "\nBuilding Graph......."
g = maxflow.Graph[float]()
nodes = g.add_nodes(N)
nodeids = nodes.reshape(h, w)
# row vector shape nodeids matrix
flatnodes = np.array(nodes)

# print flatnodes

neighdict = mut.make_neighbors(nodeids)
# Connect each pixel with its four neighbors-
# left, top, right, bottom.
# Column 0 = u , Column 1 = v, (u,v) are in nodeids
# and defines an edge: u ---> v.
# neighmat defines the grid edges
neighmat = mut.edgeconnected(nodeids)

# Make arr of input nodeids
inputnodes = []
inpulab = dict()
for i in range(len(fx)):
    nodeid = nodeids[fy[i]][fx[i]]
    inputnodes.append(nodeid)
    inpulab[nodeid] = 1.
for i in range(len(bx)):
    nodeid = nodeids[by[i]][bx[i]]
    inputnodes.append(nodeid)
    inpulab[nodeid] = 0


##################
lambdaa = 0.2
################

# Make U2
# tU2 = sum(U2) for all neighbors for each pixel
# U2dict = pairwise U2 for each pixel and its neighbor s.t-
# key = pixelid, value = {neighborid: U2neighbor)
tU2, U2dict = mut.getU2(h, w, img, neighmat, flatnodes, lambdaa)
maxwt = max(tU2)
print "Maximum U2 weight: ",maxwt

#############
# Add N-links
############
# Make U1
U1, preds = mut.getU1(img, model, neighmat, flatnodes, mu1, mu0, std1, std0)
# add N-links
visited = []
for u in U2dict:
    for v in U2dict[u]:
        edgeuv = U2dict[u][v]
        edgevu = U2dict[v][u]
        #assert edgeuv == edgevu, "false"
        g.add_edge(u, v, edgeuv, edgevu)

#################
# add T-links
#################
U1 = lambdaa*U1
for u in U2dict:
    if u in inputnodes:
        if inpulab[u] == 1:  # if u is fg user input
            g.add_tedge(u, 1+maxwt, 0)
        if inpulab[u] == 0:  # if u is bg user input
            g.add_tedge(u, 0, 1+maxwt)
    else:
        if preds[u] == 1:
            g.add_tedge(u, U1[u, 0], U1[u, 1])
        if preds[u] == 0:  # if u is not user input
            g.add_tedge(u, U1[u, 0], U1[u, 1])

# Find the maximum flow.
print "\nComputing Maxflow..."
flow = g.maxflow()
# Get the segments of the nodes in the grid.
segmat = []
for i in nodes:
    segmat.append(g.get_segment(i))

#mask = 1 - np.array(segmat)
mask = np.array(segmat)
# print np.where(mask==1)
#mask = 255*mask.reshape(h, w)
svmmask = np.array(preds, dtype='int8').reshape(h, w)
graphmask = np.int8(np.logical_not(mask.reshape(h, w)))
resgraph = cv2.bitwise_and(oimg, oimg, mask=graphmask)
ressvm = cv2.bitwise_and(oimg, oimg, mask=svmmask)
diff = np.abs(ressvm - resgraph)
print "Flow: ", flow
print "\nDone"
cv2.destroyAllWindows()
# Show the result.
cv2.namedWindow('SVM')
cv2.imshow('SVM', ressvm)
cv2.namedWindow('Graph')
cv2.imshow('Graph', resgraph)
cv2.imshow('Diff', diff)

cv2.imwrite('./results/out/' + filename + str(lambdaa)+ '_graph.jpg', resgraph)
cv2.imwrite('./results/out/' + filename + str(lambdaa) + '_svm .jpg', ressvm)
cv2.imwrite('./results/out/' + filename + str(lambdaa) + '_diff.jpg', diff)

# result = np.zeros((h, w, 3))
# for i in range(3):
#     result[:, :, i] = np.multiply(img[:, :, i], mask)

#cv2.imshow('result', mask)

# cv2.imshow('filter',img)
cv2.waitKey()
