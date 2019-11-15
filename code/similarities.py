from scipy.spatial.distance import jaccard
import numpy as np
import pandas as pd


# Computing Jaccard Distance of two 5D-Rectangles

# Issues to deal with:
# Normalizing values?
# Input format correct?
# Weighting of the different dimensions?


def jaccard_distance(datFr, name, pred):
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
    # Indexing the correct rectangles based on the name, retrieving all
    # columns, minus the "name"-one
    corr_rect = datFr.loc[datFr[0].str.match(name), 1:]
    # Computing all Jaccard Distances
    jacc_distances = corr_rect.apply(jaccard, axis=1, args=[pred])
    # Returning closest distance
    return jacc_distances.min()


"""
Returns closest Ruzicka Distance, related to Jaccard Distance, of rectangles
in the label dat and the prediction distance.

Input:
    datFr: 5 Dim. DataFrame including all labels, assuming that column 0 
        includes the names of the respective files the rectangles belong to.
    name: Name as string of the correct file.
    pred: Prediction rectangle
    
Return:
    Closest Distance (should be a float)
"""


def ruzicka_distance(datFr, name, pred):
    """
    Chooses max and min per point, ultimately returning 1 minus the sum of the 
    vector of minimal values by the sum of the vector of maximal values.
    (Ruzicka Similarity and Soergel Distance). So, if they are the same it
    returns 0, else it returns a higher value.
    """

    def ruz_similarity(x, y):
        min_vec = np.minimum(x, y)
        max_vec = np.maximum(x, y)
        # Return Soergel Distance
        return 1 - min_vec.sum() / max_vec.sum()

    # Indexing the correct rectangles based on the name, retrieving all
    # columns, minus the "name"-one
    corr_rect = datFr.loc[datFr[0].str.match(name), 1:]
    # Getting Ruzicka for all correct Rectangles
    ruz_distances = corr_rect.apply(ruz_similarity, axis=1, args=[pred])
    return ruz_distances.min()


"""
Function to incorporate both the positive and negative rectangles. Computes
both the Ruzicka distance to the closest positive and negative rectangle and 
returns the positive plus the inverted negative Soergel Distance divided by two.

Input:
    pos_df: 5 Dim. DataFrame including all labels for pos. rectangles 
        (see ruzicka_distance)
    neg_df: 5 DIm. DataFrame, but for negative rectangles
    name: Name as string of correct image
    pred: Prediction Rectangle
"""


def ruz_posneg(pos_df, neg_df, name, pred):
    ruz_pos = ruzicka_distance(pos_df, name, pred)
    ruz_neg = 1 - ruzicka_distance(neg_df, name, pred)
    return (ruz_pos + ruz_neg) / 2
