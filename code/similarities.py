from scipy.spatial.distance import jaccard
import numpy as np

# Computing Jaccard Distance of two 5D-Rectangles   

"""
Should return the "closest" jaccard distance of the rectangles in the label dat
and the prediction distance.

Input:
    datFr: 5 Dim. DataFrame including all labels, assuming that column 0 
        includes the names of the respective files the rectangles belong to.
    name: Name as string of the correct file.
    pred: Prediction rectangle
    
Return:
    Closest Distance (should be a float)
        
"""
def jaccard_distance(datFr, name, pred):
    # Indexing the correct rectangles based on the name, retrieving all
    # columns, minus the "name"-one
    corr_rect = datFr[datFr[0].str.match("name"), 1:]
    # Computing all Jaccard Distances
    jacc_distances = corr_rect.apply(jaccard, axis=1, args=[pred])
    # Returning closest distance
    return jacc_distances.min()