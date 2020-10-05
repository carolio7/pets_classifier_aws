#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @carolio7

# import des python librairies
import sys, re, subprocess
import numpy as np
import pandas as pd
import itertools

# import libraries spark
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD
from pyspark import StorageLevel

#import des librairies pour S3
import s3fs


# initialization de spark et connection en aws S3
conf = SparkConf().setAppName("Solution 2")
sc = SparkContext.getOrCreate(conf=conf)

fs = s3fs.S3FileSystem(anon=False)


# initialisation des fonctions parametrant notre programe
# Chemins AWS S3 du répertoire contenant les features en json
argument =  sys.argv[1]     # ou égal à 's3://projet2oc-rasambatra/exemple/' pour tester

# variable correspondant aux coefficients de split du dataset 
separations = np.linspace(0.0003, 0.8, 10)   #Pour varier le nombre de data dans Training set

BUCKET_DESTINATION = 's3://projet2oc-rasambatra/' # le bucket où le programme va mettre les fichieers output



# Fonctions pour notre programme

def lire_json_features(fichier):
    """
        Fonction lisant un fichier .json donné en paramètre 
        et retourne une tuple composé du nom du fichier et liste des features
            
        Attention: le repertoire doit se terminer par un slash /
    """
    label = re.sub(r'[0-9]','', fichier).split('/')[-1]     # Extraire le nom du pet a part du chemin du fichier
    label = label[:-9].strip('_')      # Enleve les extention .jpg.json au nom du fichier
    for line in fs.open('s3://' + fichier, 'r'):
                values = line.strip('[]').split(',')
                values = [float(x) for x in values]

    return (label,values)





def load_features_to_tuple(directory):
    """
    Fonction pour filtrer les fichiers .json dans un répertoire
        et retourne un rdd contenant des tuples (label du fichier, features)
            
    Attention: le repertoire doit se terminer par un slash / 
    """

    les_features_rdd = sc.parallelize(fs.ls(directory))\
                        .filter(lambda fichier: fichier[-4:] == 'json')\
                        .map(lambda les_json: lire_json_features(les_json))
    
    return les_features_rdd





def etiqueter_one_vs_all(dataset, un_label):
    """
        Fonction lisant un rdd de plusieurs tuples (label, feature):
            - convertit chaque tuple en LabelPoint 
                correspondant à la classification One-versus-All
                   » etiquette 1 si le label est égal à l'argument
                   » sinon etiquette 0 

            - retourne une liste de LabelPoint de l'argument (Label 1) versus Rest
    """

    dataset_labeled_rdd = dataset\
        .map(lambda untupl: LabeledPoint(1, untupl[1]) if (untupl[0]== un_label) else LabeledPoint(0, untupl[1]))
    
    return dataset_labeled_rdd



def etiqueter_one_vs_one(dataset, un_label, autre_label):

    """
        Fonction lisant une liste de plusieurs tuples (label, feature):
            - filte les tuples avec label correspondant aux deux arguments seulements 
            - convertit chaque tuple en LabelPoint 
                correspondant à la classification One-Versus-One
                   » etiquette 1 si le label est égal au premier argument
                   » sinon etiquette 0 pour le second

            - retourne une liste de LabelPoint du 1er arg (Label 1) versus Znd arg (Label 0)
    """

    dataset_labeled_rdd = dataset\
        .filter(lambda line: (line[0]==un_label) or (line[0]==autre_label))\
        .map(lambda untupl: LabeledPoint(1.0, untupl[1]) if (untupl[0]== un_label)\
             else LabeledPoint(0.0, untupl[1]))
    
    return dataset_labeled_rdd



def modelWithSVM(dataset_labeled, les_splits):
    """
        Entainer le modele en utilisant Support Vector Machines 
            avec different split du dataset.
        Retourne un dictionnaire du le nombre de Training set et la précision obenue
    """

    visualizationData = {}
    
    for split in les_splits:
        trainingData, validationData = dataset_labeled.randomSplit([split, 1.0-split], seed=2)
        trainingData.persist(StorageLevel.MEMORY_AND_DISK)
        validationData.persist(StorageLevel.MEMORY_AND_DISK)

        if ((trainingData.isEmpty() == False) and (validationData.isEmpty() == False)):
            model = SVMWithSGD.train(trainingData, iterations=100, step=1.0, regParam=0.01)
            predict = validationData.map(lambda ad: (ad.label, model.predict(ad.features)))
            totalValidationData = validationData.count()
            correctlyPredicted = predict.filter(lambda x: x[0] == x[1]).count()
            accuracy = float(correctlyPredicted) / totalValidationData

            visualizationData[trainingData.count()] = accuracy


        trainingData.unpersist()
        validationData.unpersist()

    return visualizationData




def apprendre_one_vs_all(dataset_rdd, label, split_list):

    """
        -Etiquetter le dataset avec le label
        -séparer avec le coefficient du split_list parcouru
        -entrainer
        Retourne un pandas dataframe dela précision obenue en fonction du nombre de Training set
        utilisé pour l'entrainement
    """

    data_labeled_ovr = etiqueter_one_vs_all(dataset_rdd, label)
    visualizationData_ovr = modelWithSVM(data_labeled_ovr, split_list)
    print('Performance : ', visualizationData_ovr)
    return pd.DataFrame(visualizationData_ovr, index=[label])



def apprendre_one_vs_one(dataset_rdd, tuple_label, split_list):

    """
        -Etiquetter le dataset avec le label
        -séparer avec le coefficient du split_list parcouru
        -entrainer
        Retourne un pandas dataframe dela précision obenue en fonction du nombre de Training set
        utilisé pour l'entrainement
    """
    
    data_labeled_ovo = etiqueter_one_vs_one(dataset_rdd, tuple_label[0], tuple_label[1])
    visualizationData_ovo = modelWithSVM(data_labeled_ovo, split_list)
    print('Performance : ', visualizationData_ovo)
    return pd.DataFrame(visualizationData_ovo, index=[tuple_label[0] + '-vs-' + tuple_label[1]])




def _write_dataframe_to_csv_on_s3(dataframe, filename):
    """ Ecrire un dataframe au format CSV sur S3 """
    print("Writing {} records to {}".format(len(dataframe), filename))
    bytes_to_write = dataframe.to_csv(None).encode()
    with fs.open(BUCKET_DESTINATION + filename, 'wb') as f:
        f.write(bytes_to_write)





print('-----------------------------------------------')
print('>>>>>>>>>>>>>>>> Début du programme')



print('»»»»»»»»»»»»»»»»»»»»»        Partie 1    ««««««««««««««««««««««««««')
print('>>>>>>>>>>>>>>> Lecture des features')


features_list_rdd = load_features_to_tuple(argument).persist(StorageLevel.MEMORY_AND_DISK)
# Extraction du nom des pets dans le dataset features_list
les_labels = features_list_rdd.keys().distinct().collect()

print("voici les labels : ")
print(les_labels)

print('>>>> Fin de lecture des features')
print('»»»»»»»»»»»»»»»»»»»»»    Fin - Partie 1    ««««««««««««««««««««««««««')






print('»»»»»»»»»»»»»»»»»»»»»    Partie 2    ««««««««««««««««««««««««««')
print('>>>> Entrainement du model SVMWithSGD One-versus-All <<<<<<<<<<<<<<<<<<<<')


result_ovr = pd.DataFrame()


for animal in les_labels:
    print('*********************   Debut - Entrainement {}-versus-All   *******************'.format(animal))

    un_dataframe_ova = apprendre_one_vs_all(features_list_rdd, animal, separations)

    result_ovr = result_ovr.append(un_dataframe_ova)
    
    print('*********************   Fin - Entrainement {}-versus-All   *******************'.format(animal))


print(' VOICI LE TABLEAU DES PRECISIONS DU MODEL :')
print(result_ovr)

print('>>>> Fin entrainement du model SVMWithSGD One-versus-All <<<<<<<<<<<<<<<<<<<<')
print('»»»»»»»»»»»»»»»»»»»»»    Fin - Partie 2    ««««««««««««««««««««««««««')






print('»»»»»»»»»»»»»»»»»»»»»    Partie 3    ««««««««««««««««««««««««««')
print('>>>> Entrainement du model SVMWithSGD One-versus-One <<<<<<<<<<<<<<<<<<<<')

result_ovo = pd.DataFrame()
# Création de couple de label deux par deux
paires = list(itertools.combinations(les_labels, r=2))
print('VOICI LES MODELS A ENTRAINER :')
print(paires)


for unePaire in paires:
    print('**********    Debut - Entrainement {}-versus-{}      ************'.format(unePaire[0],unePaire[1]))
    
    un_dataframe_ovo = apprendre_one_vs_one(features_list_rdd, unePaire, separations)

    result_ovo = result_ovo.append(un_dataframe_ovo)
    
    print('*************   Fin - Entrainement {}-versus-{}   *************'.format(unePaire[0],unePaire[1]))




print(' VOICI LE TABLEAU DES PRECISIONS DU MODEL :')
print(result_ovo)


print('>>>> Fin entrainement du model SVMWithSGD One-versus-One <<<<<<<<<<<<<<<<<<<<')
print('»»»»»»»»»»»»»»»»»»»»»    Fin - Partie 3    ««««««««««««««««««««««««««')








print('*********************    Sauvegarde des dataframes dans bucket S3      ******************************')

# Sauver les dataframes obtenus dans le bucket S3
_write_dataframe_to_csv_on_s3(result_ovr, 'output/result_ONE_VS_ALL_10_split.csv')

_write_dataframe_to_csv_on_s3(result_ovo, 'output/result_ONE_VS_ONE_10_split.csv')


print('*********************    Fin des sauvegardes      ******************************')


print('>>>>>>>>>>>>>>>>>>>>> Fin du programme')
