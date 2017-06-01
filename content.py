

import pickle as pkl
import numpy as np

AXIS_ROWS = 0
AXIS_COLUMNS = 1
CLASSES_AMOUNT = 36
NN_K = 5
DATA_AMOUNT = 27000

ACTIVATION = 3

# wczytuje dane ze zbioru - biore 6 tys egzemplarzy
"""
def get_main_data():
    x_data, y_data = pkl.load(open('train.pkl', mode='rb'))

    #x_data = x_data[0:6000]
    #y_data = y_data[0:6000]
    x_data = x_data[0:6000]
    y_data = y_data[0:6000]
    return x_data, y_data
"""

def get_main_data():
    x_data, y_data = pkl.load(open('train.pkl', mode='rb'))

    #x_data = x_data[0:6000]
    #y_data = y_data[0:6000]
    x_data = x_data[3000:DATA_AMOUNT]
    y_data = y_data[3000:DATA_AMOUNT]
    return x_data, y_data

def get_compressed_data():
    return compress_data(get_main_data())

def compress_data(data):
    x_data = data[0]
    y_data = data[1]

    return   np.array(list(map(lambda x: compress_image_x3(x, ACTIVATION), x_data))), y_data

# wybieram dane uczace, 70 procent zbioru
def get_learn_data():
    data = get_main_data()
    x_train = data[0]
    y_train = data[1]

    return (x_train)[0:int(x_train.shape[AXIS_ROWS] * 0.7)], y_train[0:int(y_train.shape[AXIS_ROWS] * 0.7)]


#wybieram dane validacyjne, 30 procent zbioru
def get_validate_data():
    x_validate = get_main_data()[0]
    y_validate = get_main_data()[1]

    return (x_validate)[int(x_validate.shape[AXIS_ROWS] * 0.7):x_validate.shape[AXIS_ROWS]], y_validate[int(
        x_validate.shape[AXIS_ROWS] * 0.7):x_validate.shape[AXIS_ROWS]]

def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    #uint16
    result = X.astype(np.uint16) @ ~(X_train.transpose().astype(bool))
    result = result + ~(X.astype(bool)) @ X_train.astype(np.uint16).transpose()
    return result

def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    return y[Dist.argsort(kind='mergesort')]

def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    columnCount = y.shape[AXIS_COLUMNS]
    resizedArray = np.delete(y, range(k, columnCount), axis=AXIS_COLUMNS)
    countedClassesOccurences = np.apply_along_axis(np.bincount, axis=1, arr=resizedArray, minlength=CLASSES_AMOUNT + 1)
    countedClassesOccurences = np.delete(countedClassesOccurences, 0, axis=1)
    #bez dzielenia, bo i tak liczy się licznik
    #probabilityOfEachClass = np.divide(countedClassesOccurences, k)

    return countedClassesOccurences

def run_program():
    validate_data = get_validate_data()
    x_valid = validate_data[0]
    y_valid = validate_data[1]
    from predict import predict
    predicted = predict(x_valid)
    print("DOBRZE PRZEWIDZIANYCH:")
    print(check_prediction(predicted,y_valid))
    """
    print("PREDICTED")
    print(predicted)
    print("VALID")
    print(y_valid)
    print()
    """

def check_prediction(result_to_chech, y_true):
    error_count = 0
    for i in range(result_to_chech.shape[0]):
        if result_to_chech[i][0] == y_true[i][0]:
            error_count += 1

    return error_count * 1.0 / result_to_chech.shape[0]


#Kompresja obrazów od Marcina:
def compress_image_x2(image, activation):
    side = int(np.sqrt(image.shape[0]))
    new_size = image.shape[0] / 4
    to_return = np.zeros(shape=new_size, dtype=np.uint8)

    def new_val(v1, v2, v3, v4):
        return 1 if v1 + v2 + v3 + v4 > activation else 0

    for a in range(28):
        for b in range(28):
            to_return[a + b * side / 2] = new_val(image[a * 2 + 2 * b * side], image[a * 2 + 1 + 2 * b * side],
                                                  image[a * 2 + 1 + (2 * b + 1) * side],
                                                  image[a * 2 + (2 * b + 1) * side])

    return to_return


def compress_image_x3(image, activation):
    side = int(np.sqrt(image.shape[0]))
    new_size = 324
    to_return = np.zeros(shape=new_size, dtype=np.uint8)

    def new_val(a, b):
        s = 0
        for aaa in range(3):
            for bbb in range(3):
                s += image[a * 3 + aaa + (b * 3 + bbb) * side]
        return 1 if s >= activation else 0

    for aa in range(18):
        for bb in range(18):
            to_return[aa + bb * 18] = new_val(aa, bb)

    return to_return


def compress_image_x4(image, activation):
    side = 56
    new_size = 196
    to_return = np.zeros(shape=new_size, dtype=np.uint8)

    def new_val(a, b):
        s = 0
        for aaa in range(4):
            for bbb in range(4):
                s += image[a * 4 + aaa + (b * 4 + bbb) * side]
        return 1 if s >= activation else 0

    for aa in range(14):
        for bb in range(14):
            to_return[aa + bb * 14] = new_val(aa, bb)

    return to_return


