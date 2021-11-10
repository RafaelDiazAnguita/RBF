#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:04 2021

IMC: lab assignment 3

@author: pagutierrez
"""

# TODO Include all neccesary imports
import pickle
import os
import numpy as np
import math
import sys
from numpy.lib.function_base import append
from pandas.core import api
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ejercicio import X_test

def checkParameters():
    parameters = []
    parameters.append(["-c",False])
    parameters.append(["-r",0.1])
    parameters.append(["-l",False])
    parameters.append(["-e",math.e-2])
    parameters.append(["-o",1])
    parameters.append(["-p",False])
    parameters.append(["-m",""])
    parameters.append(["-t",""])
    parameters.append(["-T",""])

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--help":
            print("Argumento -t, --train file: Fichero que contiene los datos de entrenamiento a utilizar.") 
            print("Argumento -T, --test file: Fichero que contiene los datos de test, si no se especifica test = train")
            print("Argumento -c, --classification: Booleano que indica si el problema es de clasificacion, Por defecto, False")
            print("Argumento -r, --ratio rbf: Indica la razon (en tanto por uno) de neuronas RBF con respecto al total de patrones en entrenamiento, Defecto 0.1")
            print("Argumento -l, --l2: Booleano que indica si utilizaremos regularizacion de L2 en lugar de la regularizacion L1, Por defecto False")
            print("Argumento -e, --eta: Indica el valor del parametro ´ eta (η). Por defecto, 1e−2.")
            print("Argumento -o, --outputs: Indica el numero de columnas de salida, Por defecto, 1.")
            print("(Kaggle) Argumento -p, --pred: Booleano que indica si utilizaremos el modo de prediccion, por defecto False")
            print("(Kaggle) Argumento -m, --model file: Indica el directorio en el que se guardaran los modelos entrenados") 
            print("Argumento --help: Mostrar la ayuda del programa ")
            exit()
        if sys.argv[i] == "-t" or sys.argv[i] == "--train_file":
            parameters[7][1] = sys.argv[i+1]
            parameters[8][1] = sys.argv[i+1]
        if sys.argv[i] == "-T" or sys.argv[i] == "--test_file":
            parameters[8][1] = sys.argv[i+1]
        if sys.argv[i] == "-c" or sys.argv[i] == "--classification":
            parameters[0][1] = True
        if sys.argv[i] == "-r" or sys.argv[i] == "--ratio_rbf":
            parameters[1][1] = float(sys.argv[i+1])
        if sys.argv[i] == "-l" or sys.argv[i] == "--la2":
            parameters[2][1] = True
        if sys.argv[i] == "-e" or sys.argv[i] == "--eta":
            parameters[3][1] = float(sys.argv[i+1])
        if sys.argv[i] == "-o" or sys.argv[i] == "--outputs":
            parameters[4][1] = int(sys.argv[i+1])
        if sys.argv[i] == "-p" or sys.argv[i] == "--pred":
            parameters[5][1] = True
        if sys.argv[i] == "-m" or sys.argv[i] == "--model_file":
            parameters[6][1] = sys.argv[i+1]

    if parameters[7][1] == "":
        print("Opción -t es necesaria.")
        exit()
    return parameters

def train_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model, pred):
    """ 5 executions of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 5 executions with different seeds.
    """

    if not pred:    

        if train_file is None:
            print("You have not specified the training file (-t)")
            return

        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)
    
        for s in range(1,6,1):   
            print("-----------")
            print("Seed: %d" % s)
            print("-----------")     
            np.random.seed(s)
            train_mses[s-1], test_mses[s-1], train_ccrs[s-1], test_ccrs[s-1] = \
                train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, \
                             model and "{}/{}.pickle".format(model, s) or "")
            print("Training MSE: %f" % train_mses[s-1])
            print("Test MSE: %f" % test_mses[s-1])
            print("Training CCR: %.2f%%" % train_ccrs[s-1])
            print("Test CCR: %.2f%%" % test_ccrs[s-1])
        
        print("******************")
        print("Summary of results")
        print("******************")
        print("Training MSE: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
        print("Test MSE: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("Training CCR: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("Test CCR: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
            
    else:
        # KAGGLE
        if model is None:
            print("You have not specified the file with the model (-m).")
            return

        # Obtain the predictions for the test set
        predictions = predict(test_file, model)

        # Print the predictions in csv format
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""            
            s += str(index)
            
            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))
                
            print(s)

def train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ One execution of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 1 executions.

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        outputs: int
            Number of variables that will be used as outputs (all at the end
            of the matrix)
        model_file: string
            Name of the file where the model will be written

        Returns
        -------
        train_mse: float
            Mean Squared Error for training data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        test_mse: float 
            Mean Squared Error for test data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        train_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
        test_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
    """
    train_inputs, train_outputs, test_inputs, test_outputs = read_data(train_file, 
                                                                        test_file,
                                                                        outputs)

    n_patterns = len(train_outputs)
    num_rbf = int(n_patterns * ratio_rbf)
    #TODO: Obtain num_rbf from ratio_rbf
    print("Number of RBFs used: %d" %(num_rbf))
    kmeans, distances, centers = clustering(classification, train_inputs, 
                                              train_outputs, num_rbf)
    
    radii = calculate_radii(centers, num_rbf)
    
    r_matrix_train = calculate_r_matrix(distances, radii)

    if classification:
        logreg = logreg_classification(r_matrix_train, train_outputs, l2, eta)
    else:
        coefficients = invert_matrix_regression(r_matrix_train, train_outputs)

    """
    TODO: Obtain the distances from the centroids to the test patterns
          and obtain the R matrix for the test set
    """
    test_distances = []
    for i in range (len(train_inputs)):
        each_distance = []
        for j in range(num_rbf):
            each_distance.append( euclidean_distance(train_inputs[i],centers[j]) )

        test_distances.append(each_distance)
    
    r_matrix_test = calculate_r_matrix(test_distances, radii)

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification' : classification,            
            'radii' : radii,
            'kmeans' : kmeans
        }
        if not classification:
            save_obj['coefficients'] = coefficients
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)
    
    # # # # # # # # # # #

    if classification:
        """
        TODO: Obtain the predictions for training and test and calculate
              the CCR. Obtain also the MSE, but comparing the obtained
              probabilities and the target probabilities
        """
        test_predictions = logreg.predict(test_inputs)
        print(test_predictions)
    else:
        """
        TODO: Obtain the predictions for training and test and calculate
              the MSE
        """
        train_predictions = np.dot(r_matrix_train, coefficients)
        test_predictions = np.dot(r_matrix_test, coefficients)
        train_mse = 0.0
        test_mse = 0.0

        train_mse = sum(pow(train_outputs-train_predictions,2)) / (outputs * len(train_outputs)) 
        test_mse = sum(pow(test_outputs-test_predictions,2)) / (outputs * len(test_outputs)) 
        
        train_ccr = 0
        test_ccr = 0


    return train_mse, test_mse, train_ccr, test_ccr

def read_data(train_file, test_file, outputs):
    """ Read the input data
        It receives the name of the input data file names (training and test)
        and returns the corresponding matrices

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        outputs: int
            Number of variables to be used as outputs
            (all at the end of the matrix).
              
        Returns
        -------
        train_inputs: array, shape (n_train_patterns,n_inputs)
            Matrix containing the inputs for the training patterns
        train_outputs: array, shape (n_train_patterns,n_outputs)
            Matrix containing the outputs for the training patterns
        test_inputs: array, shape (n_test_patterns,n_inputs)
            Matrix containing the inputs for the test patterns
        test_outputs: array, shape (n_test_patterns,n_outputs)
            Matrix containing the outputs for the test patterns
    """
    #TODO: Complete the code of the function
    train_df = pd.read_csv(train_file,header=None)

    train_inputs = train_df.values[:,0:-outputs]
    train_outputs = train_df.values[:,-outputs]

    test_df = pd.read_csv(test_file,header=None)

    test_inputs = test_df.values[:,0:-outputs]
    test_outputs = test_df.values[:,-outputs]

    return train_inputs, train_outputs, test_inputs, test_outputs

def init_centroids_classification(train_inputs, train_outputs, num_rbf):
    """ Initialize the centroids for the case of classification
        This method selects in a stratified num_rbf patterns.

        Parameters
        ----------
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        centroids: array, shape (num_rbf,n_inputs)
            Matrix with all the centroids already selected
    """
    
    #TODO: Complete the code of the function
    return centroids

def euclidean_distance(a,b):

    sumatory = 0
    for k in range(len(a)):
        sus = a[k] - b[k]
        sumatory += pow(sus,2)
    return math.sqrt(sumatory) 

def clustering(classification, train_inputs, train_outputs, num_rbf):
    """ It applies the clustering process
        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification(). 
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        centers: array, shape (num_rbf,n_inputs)
            Centers after the clustering
    """

    #TODO: Complete the code of the function
    kmeans = KMeans(n_clusters = num_rbf,max_iter = 500,n_init=1)

    if classification:
        index = np.random.choice(train_inputs.shape[0],num_rbf,replace=False)
        patterns = train_inputs[index]
        kmeans.fit(patterns)
    else:
        index = np.linspace(0,len(train_inputs)-1,num_rbf)
        patterns = train_inputs[index.astype(int)]        
        kmeans.fit(patterns)
    

    centroids = kmeans.cluster_centers_
    
    distances = []
    for i in range (len(train_inputs)):
        each_distance = []
        for j in range(num_rbf):
            each_distance.append( euclidean_distance(train_inputs[i],centroids[j]) )

        distances.append(each_distance)
        
    
    return kmeans, distances, centroids

def calculate_radii(centers, num_rbf):
    """ It obtains the value of the radii after clustering
        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Parameters
        ----------
        centers: array, shape (num_rbf,n_inputs)
            Centers from which obtain the radii
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
    """

    #TODO: Complete the code of the function
    radii = []
    for i in range(len(centers)):
        sum = 0
        for j in range(len(centers)):
            if i != j:
                sum += euclidean_distance(centers[i],centers[j])
        radii.append( sum / (2*(num_rbf-1)) )

    return radii

def calculate_r_matrix(distances, radii):
    """ It obtains the R matrix
        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias
        
        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
            
        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
    """

    #TODO: Complete the code of the function
    r_matrix = []
    for i in range(len(distances)):
        outs = []
        for j in range(len(distances[i])):
            outs.append(pow(math.e, -distances[i][j] / (2*pow(radii[j],2) ) ) )
        outs.append(1)
        r_matrix.append(outs)

    return r_matrix

def invert_matrix_regression(r_matrix, train_outputs):
    """ Inversion of the matrix for regression case
        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
              
        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value 
            of the bias 
    """

    #TODO: Complete the code of the function
    R = np.array(r_matrix)
    B = np.array(train_outputs)
    R1 = np.linalg.pinv(R)
    coefficients = np.matmul(R1,B)

    return coefficients

def logreg_classification(matriz_r, train_outputs, l2, eta):
    """ Performs logistic regression training for the classification case
        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
              
        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression
            Scikit-learn logistic regression model already trained
    """

    #TODO: Complete the code of the function
    if (l2):
        logreg = LogisticRegression( C=1/eta,penalty='l2',solver='liblinear')
    else:
        logreg = LogisticRegression( C=1/eta,penalty='l1',solver='liblinear')

    logreg.fit(matriz_r,train_outputs)
    return logreg


def predict(test_file, model_file):
    """ Performs a prediction with RBFNN model
        It obtains the predictions of a RBFNN model for a test file, using two files, one
        with the test data and one with the model

        Parameters
        ----------
        test_file: string
            Name of the test file
        model_file: string
            Name of the file containing the model data

        Returns
        -------
        test_predictions: array, shape (n_test_patterns,n_outputs)
            Predictions obtained with the model and the test file inputs
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]
    
    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    radii = saved_data['radii']
    classification = saved_data['classification']
    kmeans = saved_data['kmeans']
    
    test_distancias = kmeans.transform(test_inputs)    
    test_r = calculate_r_matrix(test_distancias, radii)    
    
    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coefficients']
        test_predictions = np.dot(test_r, coeficientes)
        
    return test_predictions
    
if __name__ == "__main__":
    parameters = checkParameters()
    parameters
    train_rbf_total(parameters[7][1],parameters[8][1],parameters[0][1],parameters[1][1],parameters[2][1],parameters[3][1],
                        parameters[4][1],parameters[6][1],parameters[5][1])
