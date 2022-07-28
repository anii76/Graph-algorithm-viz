from token import OP
import numpy as np
import random
# import branchNbound as bnb
import time
from random import randrange
import math
from operator import countOf

logfile = 'colorslist.txt'

#from TABUPPA import TabuSearch
nbr_iter = 20
max_init_heur = 1
population_size = 8
lambda_max = 15 # the maximum step length a prey can run
lambda_min = 1  # the minimum step length a prey can run
P = 0.5
#n = 0.1
B = 0.5
k = 30
tau=0.33

class Graph(object):

    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1

    # Remove edges
    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        for row in self.adjMatrix:
            for val in row:
                print('')
            print(self.adjMatrix)


def voisins(G, i):
    M = G.adjMatrix
    listeVoisins = []
    n = len(M)
    for j in range(n):
        if M[i][j] == 1:
            listeVoisins.append(j)
    return (listeVoisins)


def conflict(G,
             c):  # fonction qui donne le nombre et les sommets en conflits  de coleurs dans un graphe G avec un coloration donnée
    listConflicts = []
    nbConf = 0
    for v in range(len(c)):
        for j in voisins(G, v):
            if c[v] == c[j]:
                listConflicts.append(v)
                nbConf += 1
    return [np.unique(listConflicts), nbConf / 2]


###################################################################################################################
############lambdas, P, n, B, k and w are algo parameters#########


# define survival function as the reverse of obj function = 1/nb_used_colors
def max_vertex_degree(G):
    # trier les sommets de G selon l'ordre décroissant de leurs degrés
    graph = G.adjMatrix
    degrees = [[j, sum(i)] for i, j in zip(graph, range(len(G)))]
    degrees.sort(key=lambda degrees: degrees[1])
    degrees.reverse()
    return degrees[0][1]  # degré maximal


def generate_random_coloring(G, max_col):
    M = G.adjMatrix
    n = len(M)
    coloring = [-1 for j in range(n)]
    for i in range(n):
        unsafe = True
        while unsafe:
            color = random.randint(0, max_col)
            neighbors = voisins(G, i)
            forbiden = [coloring[v] for v in neighbors]
            if not (color in forbiden):
                unsafe = False
                coloring[i] = color

    return coloring


def generate_p(G, nbre, max_col):
    population = []
    for i in range(nbre):
        s = generate_random_coloring(G, max_col)
        population.append(s)
    return np.array(population)


def movePredator(xpred, X, SV):
    epsilon1 = random.random()
    epsilon2 = random.random()
    xprey = X[0]
    xprey = np.array(xprey)
    # find the prey with the least surviving value
    for xi in range(len(X)):
        if SV[xi] <= 1 / len(np.unique(xpred)):
            xprey = np.array(X[xi])

    # initialize the random direction
    yr = np.random.rand(1, len(xprey))
    xpred = xpred + (lambda_max * epsilon1 * (yr / np.linalg.norm(yr)) + lambda_min * epsilon2 * (
            (xprey - xpred) / np.linalg.norm(xprey - xpred))).astype(int)
    # print('xpred',list(xpred[0]))
    return xpred[0]


def movePrey(G, xpred, xi, X, SV, k):
    A = []
    U = []
    Y = []
    Z = []
    SVZ = []
    epsilon1 = random.random()
    epsilon2 = random.random()
    epsilon3 = random.random()
    epsilon4 = random.random()
    # find the preys with better surviving values than xi
    for xj in range(len(X)):
        if SV[xi] <= SV[xj]:
            A.append(X[xj])
    # if xi is the prey with the best surviving value
    if len(A) == 0:
        # generate random k unit directions
        for j in range(k):
            U.append(np.random.rand(1, len(X[xi])))
            Z[j] = X[xi] + (lambda_min * epsilon1 * U[j]).astype(int)
        Z[k + 1] = X[xi]  # U[k+1]=0
        for Zj in range(len(Z)):  # compute survival value for each Zj
            SVZ[Zj] = 1 / (len(np.unique(Z[Zj])))
        Zl = 0
        # find the prey with the best surviving value and update xi
        for Zj in range(len(Z)):
            if SVZ[Zj] >= SVZ[Zl]:
                Zl = Zj
        xi = Zl
    else:
        r = random.random()
        if r <= P:
            # calculer Yi
            yi = np.zeros(len(X[xi]))
            for xj in range(len(A)):
                # calculer la distance euclidienne entre xi et xj
                rij = np.linalg.norm(np.array(A[xj]) - np.array(X[xi]))
                #yi = yi + (n ** SV[xj] * np.subtract(X[xj], X[xi])) / rij
               #equation 3.3
                arr=(np.exp(SV[xj]**tau-rij))*np.subtract(X[xj], X[xi])
                yi=np.sum(arr)
            # generate random k unit directions
            for j in range(k):
                Y.append(np.random.rand(1, len(X[xi])))
                Z.append(X[xi] + (lambda_min * Y[j]).astype(int))
            # update yr
            Zl = 0
            for Zj in range(len(Z)):
                if (1 / len(np.unique(X[xi] + (lambda_min * yi).astype(int)))) >= (
                        1 / len(np.unique(Z[Zl]))):  # compute survival value for each Zj
                    Zl = Zj
            yr = yi
            # update xi
            lmax = lambda_max * (math.exp(B * abs(SV[xi] - 1 / len(np.unique(xpred)))))
            X[xi] = X[xi] + (lmax * epsilon2 * (yi / np.linalg.norm(yi)) + epsilon3 * (yr / np.linalg.norm(yr))).astype(int)  # a changer
        else:
            # generate a random unit direction yr
            yr = np.random.rand(1, len(X[xi]))
            # calculate d1 and d2
            d1 = np.linalg.norm(xpred - (xi + yr))
            d2 = np.linalg.norm(xpred - (xi - yr))
            if d1 <= d2:
                yr = -yr
            X[xi] = X[xi] + (lambda_max * epsilon4 * yr).astype(int)
   # print("xi                                                ", list(X[xi]))
    return X[xi]


def PPA(G, population_size, max_col,nbr_iter):
    X = generate_p(G, population_size, max_col)
    OptSol=[]
    for i in range(nbr_iter):
       # print('iteration', i)
        SV = [0 for i in range(len(X))]
        for x in range(len(X)):  # compute survival value for each sol
            SV[x] = 1 / (len(np.unique(X[x])))
        # xpred=the sol with the least SV
        xpredi = 0
        for x in range(len(X)):
            if SV[x] <= SV[xpredi]:
                xpredi = x

        xpred = X[xpredi]

        #X.remove(xpred)  # remove the predator from the population
        X = np.delete(X, xpredi, 0)
        for xi in range(len(X)):
            X[xi] = movePrey(G, xpred, xi, X, SV, k)
        xpred = movePredator(xpred, X, SV)
        X = np.append(X, [xpred], axis=0)
        #X.append(xpred)
        # return the best prey in the population
        xopt = 0
        for x in range(len(X)):
            if SV[x] >= SV[xopt]:
                xopt = x
        OptSol = X[xopt]
        if min(OptSol) < 0:
            OptSol = OptSol + abs(min(OptSol))
   # print('ggggggggggggg',len(np.unique(OptSol)))

        # Update color values at each iteration
        if (not any(n < 0 for n in list(OptSol))):
            with open(logfile, "a") as file:
                file.write(str(list(OptSol))+"\n")
        #print("colors list",list(OptSol))
   
    #OptSol,nb=TabuSearch(G,10,10,OptSol,5)

    # change negative colors
    for i in range(len(OptSol)):
        if OptSol[i] < 0:
            change(G, i, OptSol)
    # validate the best solution by choosing a color that's not assigned to the neighbors of each vertex in conflict
    nbConf = conflict(G, OptSol)[1]  # number of conflicts in the best solution found so far
    if nbConf != 0:
        nodeConf = conflict(G, OptSol)[0]  # list of conflicting nodes
        for r in nodeConf:  # while there are still conflicting nodes
            # SI CONFLIT ON TROUVE COLEUR QUI N EST PAS
            change(G, r, OptSol)
   # print("une coloration est trouvée: ", OptSol, "avec ", len(np.unique(OptSol)), " couleurs")
    return OptSol, len(np.unique(OptSol))
    
def change(G, sommet, c):
    vois = voisins(G, sommet)
    couleurmax = max(c)
    unique = np.unique(c)

    col_voisin = [c[i] for i in vois]
    col_used_by_neigb = np.unique(col_voisin)
    poss = [c for c in unique if c not in col_used_by_neigb and c > 0]
    if len(poss):
        c[sommet] = poss[0]
    else:
        c[sommet] = couleurmax + 1
    return c

def main_ppa(benchmark: str, results: list, nbr_col_max):

    print('----- ppa running----')
    M = np.genfromtxt(benchmark)
    M = M[:, 1:]
    y = []
    for i in M: 
        y.append(int(i[0]))
        y.append(int(i[1]))

    nb_sommets = max(y)
    g = Graph(nb_sommets)
    for i in M:
        g.add_edge(int(i[0]) - 1, int(i[1]) - 1)


    #make sure file is empty
    f = open(logfile, 'w')
    f.close()
    

    start_time = time.time()
    population_size=10 #fixe
    #nbr_col_max=15 # 
    test=1
    time_best_sol=0
    choosen=[]
    minim=nbr_col_max
    '''for population_size in range(3,25,5):
        minim=nbr_col_max'''
    #for i in range(test):
    start_time = time.time()
    sol,nbre_color_used= PPA(g, population_size,nbr_col_max,nbr_iter)
    end_time = time.time()
    ExecTime = end_time - start_time
    minim=nbre_color_used
    choosen=sol
    
    print("une coloration est trouvée: ", sol, "avec ", len(np.unique(choosen)), " couleurs")
    print("Temps d'exécution en secondes = ", ExecTime)

    results.append(ExecTime)


    ''' if nbre_color_used<minim:
                minim=nbre_color_used
                choosen=sol
                time_best_sol=ExecTime'''
    #f = open("testPPA.txt", "a")
    #f.write("\n"+"most important parameters  are \n")
    #f.write("population_size: "+str(population_size)+"\n")
    #f.write("number of iterations: "+str(nbr_iter )+"\n")
    #f.write("follow_up propability: "+str(P)+"\n")
    #f.write("lamdamin: "+str(lambda_min)+"\n")
    #f.write("lambdamax: "+str(lambda_max)+"\n")
    #f.write("solution is " + str(choosen) +"\n")
    #f.write("colors used: "+str(minim)+"\n")
    #f.write("execution time : "+str(time_best_sol)+"\n")
    #f.close()



#main()
