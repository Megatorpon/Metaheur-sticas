#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:57:15 2020

@author: megatorpon
"""

import numpy as np
import csv
import time
import sys
import getopt

## Funciona correctamente
def euclidean_distance(a, b):
    n = len(a)
    euclidean_dist = 0
    
    for i in range(n):
        euclidean_dist += (a[i]-b[i])*(a[i]-b[i])
        
    euclidean_dist = np.sqrt(euclidean_dist)
    
    return euclidean_dist

class PAR:
    
    def __init__(self, dataset_file, restraints_file):
        self.X = None
        self.data_size = 0
        self.k = None
        self.clusters = None
        self.R_1 = None
        self.R_1_list = []
        self.dimensions = None

        self.read_data(dataset_file, restraints_file)
        self.processed_X = np.zeros((len(self.X)), dtype=bool)
        self.centroids = np.zeros((self.k, self.dimensions)) 

    def compute_centroid (self, cluster):
        
        centroid = np.zeros(self.dimensions)
    
        if (len(cluster) > 0):
            for index in cluster:
                centroid += self.X[index]
                
            centroid = np.divide(centroid, len(cluster))
                   
        return centroid
        
    def compute_intra_cluster_distance(self, cluster_index):
        intra_cluster_distance = 0
        
        for data_index in self.clusters[cluster_index]:
            intra_cluster_distance += euclidean_distance(self.X[data_index], 
                                                         self.centroids[cluster_index])
            
        intra_cluster_distance = np.divide(intra_cluster_distance, 
                                           len(self.clusters[cluster_index]))
        
        return intra_cluster_distance
    
    
    # Parámetros: Índice del elemento y el cluster en el que desea meter
    # (el vector entero, no el índice del cluster)
    # Funciona correctamente
    def infeasibility (self, element_index, cluster):
        
        infeasibility = 0
        
        if (self.R_1.size != 0):            
            for index, restraint in enumerate(self.R_1[element_index]):
                
                if (restraint != 0 and 
                    self.processed_X[index] == True and
                    index != element_index):
                        
                    i = np.where(cluster==index)
                    
                    if (restraint == 1 and i[0].size == 0):
                        infeasibility += 1
                            
                    elif (restraint == -1 and i[0].size != 0):
                        infeasibility += 1
        else:
            infeasibility = -1
        
        return infeasibility


    def compute_general_infeasibility(self, S):
    
        general_infeasibility = 0
        """
        for element_index, c_index in enumerate(S):
    
            restraint_index = element_index + 1
            cluster = np.array(self.clusters[S[element_index]])
            
            while(restraint_index < self.data_size):
                restraint = self.R_1[element_index][restraint_index]
                
                if (restraint != 0):
                    
                    i = np.where(cluster==restraint_index)
                    
                    if (restraint == 1 and i[0].size == 0):
                        general_infeasibility += 1
                            
                    elif (restraint == -1 and i[0].size != 0):
                        general_infeasibility += 1
            
                restraint_index += 1
        """
        
        for i, restraint in enumerate(self.R_1_list):
            
            c_1 = int(S[restraint[0]])
            c_2 = int(S[restraint[1]])
            rest_type = int(restraint[2])
            
            if (rest_type == 1 and c_1 != c_2):
                general_infeasibility += 1
                
            elif (rest_type == -1 and c_1 == c_2):
                general_infeasibility += 1
                
        return general_infeasibility 
        
    def general_deviation(self):
        deviation = 0
        
        for i in range(self.k):
            deviation += self.compute_intra_cluster_distance(i)
        
        deviation = np.divide(deviation, self.k)
        
        return deviation
    
    def general_deviation_S(self, S):
        
        deviation = 0
            
        self.initialize_clusters()
        
        for element_index, c_index in enumerate(S):
            self.clusters[c_index].append(int(element_index))
            
        for i in range(self.k):
            self.centroids[i] = self.compute_centroid(self.clusters[i]) 
            
        for i in range(self.k):
            deviation += self.compute_intra_cluster_distance(i)
            
        deviation = np.divide(deviation, self.k)
        
        return deviation
            
    # Funciona correctamente
    def read_data(self, dataset_file, restraints_file):
        
        if (dataset_file == 'iris_set.dat' or dataset_file == 'rand_set.dat'):
            self.k = 3
            
        elif (dataset_file == 'ecoli_set.dat'):
            self.k = 8
        
        else:
            raise Exception('The dataset file passed as argument is invalid. '
                            'The value was: {}'.format(dataset_file))
            
        with open(dataset_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            local_X = []
            for i, row in enumerate(csv_reader):
                local_X.append(row)
                    
            self.X = np.empty((len(local_X), len(local_X[0])))
            
            for i, row in enumerate(local_X):
                for j, data in enumerate(row):
                    self.X[i][j] = data
            
            self.data_size = csv_reader.line_num

        self.R_1 = np.zeros((self.data_size, self.data_size))
        self.R_2 = np.zeros((self.data_size, self.data_size))
        
        with open(restraints_file) as csv_file:
            
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            for i, row in enumerate(csv_reader):
                for j, res in enumerate(row):
                    self.R_1[i][j] = int(res)
                    
                    if (int(res) != 0 and j > i):
                        restraint = [i, j, res]
                        self.R_1_list.append(restraint)
                        
        self.dimensions = len(self.X[0])
        
    def initialize_centroids(self):
        min_limits = self.X[0].copy()
        max_limits = self.X[0].copy()
        
        for i, row in enumerate(self.X):
            for j, data in enumerate(row):
                
                if (min_limits[j] > data):
                    min_limits[j] = data
                    
                if (max_limits[j] < data):
                    max_limits[j] = data
    
        # Inicializamos los centroides aleatoriamente
        
        for i, c in enumerate(self.centroids):
            for j, data in enumerate(c):
                c[j] = np.random.uniform(min_limits[j], max_limits[j])
          
            
    def are_there_empty_clusters(self, S):
        empty_clusters = False
        
        for i in range(self.k):
            count = np.count_nonzero(S == i)
            
            if (count == 0):
                empty_clusters = True
                break
            
        return empty_clusters
    
    def compare_vectors(self, a, b):
        
        is_equivalent = None
        
        if (len(a) == len(b)):
            is_equivalent = True
            
            for i in range(len(a)):
                for j in range(len(a[0])):
                    if (a[i][j] != b[i][j]):
                        is_equivalent = False
                        break
                
                if (is_equivalent == False):
                    break
            
        
        else:
            is_equivalent = False
            
        return is_equivalent
        
    # Funciona correctamente
    def initialize_clusters(self):
         
        self.clusters = []
        for i in range(self.k):
            self.clusters.append([])
                
    def compute_COPKM(self):
        RSI = [*range(len(self.X))]
        np.random.shuffle(RSI)
        general_infeasibility = None

        centroids_copy = np.ones((self.k, self.dimensions))
        while ((self.compare_vectors(centroids_copy, self.centroids)) == False):
            
            general_infeasibility = 0
            centroids_copy = self.centroids.copy()
            self.initialize_clusters()
            self.processed_X = np.zeros(self.data_size, dtype=bool)
            
            for y, index in enumerate(RSI):
                
                infeasibility_by_cluster = []
                
                for c in self.clusters:
                    inf = self.infeasibility(index, np.array(c))
                    infeasibility_by_cluster.append(inf)
                    
                min_inf = min(infeasibility_by_cluster)
                general_infeasibility += min_inf
                ind = infeasibility_by_cluster.index(min_inf)
                count = infeasibility_by_cluster.count(min_inf)
                
                if (count == 1):
                    self.clusters[ind].append(index)
                    
                elif count > 1:
                    
                    min_dist = 10000.0
                    min_dist_index = ind
                    
                    for i, inf in enumerate(infeasibility_by_cluster):
                        
                        if (inf == min_inf):
                            dist = euclidean_distance(self.X[index], 
                                                      self.centroids[i])
                            
                            if (min_dist > dist):
                                min_dist = dist
                                min_dist_index = i

                    self.clusters[min_dist_index].append(index)
                    
                self.processed_X[index] = True
            
            for i in range(self.k):
                self.centroids[i] = self.compute_centroid(self.clusters[i])
                #print("Cluster", i, "\n", self.clusters[i], "\n")            
            
        for c in self.clusters:
            print("min:", min(c), "\nmax:", max(c), "\nsize:", len(c))
            print(c, "\n")
            
        print("\nInfeasibilidad general final", general_infeasibility)
        print("Desviacion general final", self.general_deviation())
        
    # Funciona correctamente
    def Cambio_cluster(self, S, i, l):
        S[i] = l
    
    # Funciona correctamente (testeado con Luis)
    def compute_max_distance(self):
        max_distance = 0
        
        for i in range(self.data_size):
            j = i + 1
            while (j < self.data_size):
                local_distance = euclidean_distance(self.X[i], self.X[j])
                
                if (max_distance < local_distance):
                    max_distance = local_distance
                    
                j += 1

        return max_distance  
         
    def compute_objective_function(self, S, _lambda):
        
        d = self.general_deviation_S(S)
        infeasibility = self.compute_general_infeasibility(S)
        f = d + (infeasibility * _lambda)
        
        #print(d, "+", infeasibility, "=", f)
        return f
    
    def compute_local_search(self):
        
        # Se inicializan los clusters, dejándolos vacíos
        self.initialize_clusters()
        
        # Inicializamos S aleatoriamente
        S = np.zeros(self.data_size, dtype=int)
        while (True):
            
            clusters_used = np.zeros((self.k), dtype=bool)
            
            for element_index in range(len(S)):
                cluster_index = np.random.randint(0, self.k)
                S[element_index] = cluster_index
                
                if (clusters_used[cluster_index] == False):
                    clusters_used[cluster_index] = True
                
            if (self.are_there_empty_clusters(S) == False):
                break;
            
        # El vector de procesados lo ponemos entero a true
        self.processed_X = np.ones((self.data_size), dtype=bool)
        
        EVALUATION_LIMIT = 100000
        best_upon_neighbors = False
        
        # Calculamos lambda
        _lambda = np.divide((self.compute_max_distance())*3.0, len(self.R_1_list))
        current_eval = self.compute_objective_function(S, _lambda)
        evaluations = 1
        
        # Mientras no se llegue al límite de evaluaciones y
        # mientras se encuentre un vecino mejor al evaluado
        while (evaluations < EVALUATION_LIMIT and 
               best_upon_neighbors == False):
            
            best_upon_neighbors = True
            
            # Creamos las parejas índice-valor que nos servirán para
            # establecer los vecinos
            neighbors = []  
            for i, c_index in enumerate(S):
                for j in range(self.k):
                    if (c_index != j):
                        n = [i, j]
                        neighbors.append(n)
            
            # Los barajamos
            np.random.shuffle(neighbors)
            S_copy = None
            
            # Recorremos todos los vecinos para ver si alguno nos da un valor
            # de la función objetivo menor al actual
            for index, n in enumerate(neighbors):
                S_copy = S.copy()
                
                self.Cambio_cluster(S_copy, n[0], n[1])
                
                empty_clusters = self.are_there_empty_clusters(S_copy)
                
                if (empty_clusters == False):
                    neighbor_eval = self.compute_objective_function(S_copy, _lambda)
                    evaluations += 1
                    
                    if (neighbor_eval < current_eval):
                        #print(neighbor_eval, "<", current_eval)
                        self.Cambio_cluster(S, n[0], n[1])
                        current_eval = neighbor_eval
                        best_upon_neighbors = False
                        break
                            
                    if (evaluations == EVALUATION_LIMIT):
                        break
            
        print("Evaluations:", evaluations)    
        print("Desviación general:", self.general_deviation_S(S))
        print("Infeasibility:", self.compute_general_infeasibility(S))
        print("Agregado final:", self.compute_objective_function(S, _lambda))
                
        return S
            
"""
_____________________________________________
Lectura de ficheros
_____________________________________________

"""

def main(argv):      
    dataset = ''
    restriction = ''
    algorithm = ''
    output_file = ''
    
    try:
        opts, args = getopt.getopt(argv, "d:r:a:o:", [])
    except getopt.GetoptError:
        print('python PAR.py -d <data_file> -r <restriction_file> -a <algorithm>'
              ' -o <output_file>')
        sys.exit(2)
                
    for opt, arg in opts:
        if opt == '-d':
            dataset = arg
        elif opt == '-r':
            restriction = arg
        elif opt == '-a':
            algorithm = arg
        elif opt == '-o':
            output_file = arg
            
    if (dataset == '' or restriction == '' or algorithm == ''):
        raise Exception('Usage: python PAR.py -d <iris/ecoli/rand>'
                        '-r <10/20> -a <COPKM/BL> -o <output_file>')
        
    if (output_file != ''):
        sys.stdout = open(output_file, 'w')
        
    dataset_file = dataset + "_set.dat"
    restriction_file = dataset + "_set_const_" + restriction + ".const"
    PAR_object = PAR(dataset_file, restriction_file)
    print ("Dataset:", dataset, "\nRestrictions:", 
           restriction + "%", "\nAlgorithm:", algorithm)
    
    
    if (algorithm == 'COPKM'):
        i_time = time.time_ns() / (10 ** 9)
        PAR_object.compute_COPKM()
        f_time = time.time_ns() / (10 ** 9)
        print('Elapsed Time: ' + str(f_time - i_time))  
    
    if (algorithm == 'BL'):
        i_time = time.time_ns() / (10 ** 9)
        PAR_object.compute_local_search()
        f_time = time.time_ns() / (10 ** 9)
        print('Elapsed Time: ' + str(f_time - i_time))  
        
        print("Salgo del bucle general")
    
if __name__ == "__main__":
    main(sys.argv[1:])