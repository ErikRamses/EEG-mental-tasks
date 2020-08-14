#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from scipy.signal import wiener, lfilter
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
#import mne

# Definition of channel types and names.
#sfreq = 2500  # Sampling frequency
#ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog']
#ch_names = ['c3', 'c4', 'p3', 'p4', 'o1', 'o2', 'eog']

#info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

class EegSignal(object):
    def __init__(self, subject, task, channels, data, coefs=None):
        if coefs is None:
            coefs = []
        self.subject = subject
        self.task = task
        self.trial = channels
        self.data = data
        self.length = coefs
        self.energy = coefs


def convm(x, p):
    n = len(x) + 2 * p - 2
    new_x = []
    x = np.asarray(x)
    xpad = np.concatenate([np.zeros(p - 1), x, np.zeros(p - 1)])
    for i in range(1, p + 1):
        new_x.append(xpad[p - i: n - i + 1])
    new_x = np.matrix(new_x)
    return new_x


# Get covariance hayes function
def covar(x, p):
    m = len(x)
    x = x - (np.ones(m) * (sum(x) / m))
    mat = convm(x, p)
    r = mat * (mat.T / (m - 1))
    return r


def le_transform(x):
    m, n = x.shape
    y = []
    q = 5
    padding = np.zeros(q-1)
    for ii in range(m):
        y.append(np.concatenate([padding, x[ii]]))
    y = np.asarray(y)
    l = []
    e = []
    for i in range(n):
        length = 0
        energy = 0
        for k in range(i, i + q - 1):
            delta = 0
            for j in range(m):
                if k == 0:
                    delta = 0
                else:
                    delta += (y[j][k + 1] - y[j][k]) ** 2.0
            length += delta ** (1.0 / 2.0)
            energy += delta
        l.append(length)
        e.append(energy)
    l = np.asarray(l)
    e = np.asarray(e)
    return l, e


# Obtención del modelo autorregresivo de orden p
def autoreg_model(X, p):
    model = AR(X)
    model_fit = model.fit(p)
    return model_fit.params

# Concatenar canales con modelo autorregresivo
def six_channels_one(X, p):
    trial = np.array([])
    for i in range(len(X)):
        trial = np.concatenate([trial, autoreg_model(X[i], p)])
    return trial

# Remover Ruido Con Eog data
def removeArtifact(data):
    p = 2
    eog = data[6]
    Rv2 = covar(eog, p)

    signal = np.array([data[0], data[1], data[2], data[3], data[4], data[5]])
    for i in range(len(signal)):
        rxv2 = convm(signal[i], p) * convm(eog, p).T / (len(signal[i]) - 1)
        w = rxv2[0] * np.linalg.inv(Rv2)
        v1hat = lfilter(w.A1, 1, eog)
        data[i] = signal[i] - v1hat
    return data


# Método para el cálculo del mejor orden
def orderSelection(a):
    model = AR(a)
    maxlag = 28
    order = model.select_order(maxlag=maxlag, ic='aic', trend='nc')
    return order

def distance_malahanobis(xx, classes):
    distances = []
    p = len(xx)
    for i in range(len(classes)):
        iv = np.linalg.inv(covar(classes[i], p))
        results = distance.mahalanobis(xx, classes[i], iv)
        distances.append(results)
    return distances

def distance_euclidean(xx, classes):
    distances = []
    for i in range(len(classes)):
        results = np.linalg.norm(xx - classes[i])
        distances.append(results)
    return distances

def getAverage(eeg):
    task_avg = eeg[0].length
    for i in range(1, len(eeg)):
        task_avg += eeg[i].length
    task_avg = task_avg / len(task_avg)
    return np.asarray(task_avg)

def getClassAverages(eeg, tasks):
    #classes_e = []
    classes_l = []
    for i in range(len(tasks)):
        #classes_e.append([])
        classes_l.append([])

    for item in eeg:
        index = tasks.index(item.task)
        #classes_e[index].append(item.energy)
        classes_l[index].append(item.length)

    #class_avg_e = []
    class_avg_l = []

    for i in range(len(classes_l)):
        #class_avg_e.append(np.mean(classes_e[i], axis = 0))
        class_avg_l.append(np.mean(classes_l[i], axis = 0))
    return class_avg_l

def confusion_matrix(a, b, target):
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(len(a)):
        if a[i] == target:
            if a[i] == b[i]:
                tp+=1
            else:
                fn+=1
        else:
            if b[i] == target:
                fp+=1
            else:
                tn+=1
    return tn, fp, fn, tp



mat = scipy.io.loadmat('eegdata.mat')
eeg = []

subjects = []
tasks = []
trials = []
orders = []

items = mat['data'][0]

items = np.delete(items, [164])
for trial in items:
    if(trial[0][0][0] == 'subject 5'):
        eeg.append(EegSignal(trial[0][0][0], trial[0][1][0], trial[0][2][0], trial[0][3]))
    tasks.append(trial[0][1][0])
    subjects.append(trial[0][0][0])
    trials.append(trial[0][2][0])

tasks = list(dict.fromkeys(tasks))
subjects = list(dict.fromkeys(subjects))
trials = list(dict.fromkeys(trials))

order = 23

for i in range(len(eeg)):
    # Remover artefactos con fuente externa EOG
    eeg[i].data = removeArtifact(eeg[i].data)
    eeg[i].length = six_channels_one(eeg[i].data, order)

class_avg = getClassAverages(eeg, tasks)

distances = []
for i in range(len(eeg)):
    eeg[i].length = distance_malahanobis(eeg[i].length, class_avg)

original = []
predicted = []

for i in range(len(eeg)):
    original.append(tasks.index(eeg[i].task))
    predicted.append(eeg[i].length.index(min(eeg[i].length)))

print(original, predicted)

for i in range(len(class_avg)):
    print(tasks[i], confusion_matrix(original, predicted, i))
    plt.plot(class_avg[i], label=tasks[i])
plt.title('Senales')
plt.ylabel('Magnitud')
plt.xlabel('Frecuencia')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()