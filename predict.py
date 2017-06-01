# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from content import get_learn_data, hamming_distance, NN_K, sort_train_labels_knn, p_y_x_knn, AXIS_COLUMNS


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    x_train = get_learn_data()[0]
    y_train = get_learn_data()[1]

    distances_array = hamming_distance(x, x_train)
    sorted_labels = sort_train_labels_knn(distances_array, y_train)
    each_class_probability = p_y_x_knn(sorted_labels, 1)
    predicted_vector = np.argmax(each_class_probability,axis=AXIS_COLUMNS) + 1
    #add 1 to all vector class
    return predicted_vector