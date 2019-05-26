import os
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
from utilities import *
from scipy.spatial import distance
from audiotsm.io.array import ArrayReader, ArrayWriter
from audiotsm import wsola
from dtw import dtw
from collections import defaultdict

# False corresponds to Model 1
# True corresponds to Model 2
DETECT_ENDPOINT = False

# For each Model, m and s have been precalculated to save time in successive executions
if not DETECT_ENDPOINT:
    m = 14.75910
    s = 1.060054
else:
    m = 23.04372578909789
    s = 3.9701596462434754
print('Threshold =', m+3*s)


def parse(test, train, outcome):
    '''Returns whether the prediction was True Positive, False Negative, False Positive or False negative'''
    if (test.name, test.word) != (test.claim, train.word):
        if outcome:
            return 'FP'
        else:
            return 'TN'
    else:
        if outcome:
            return 'TP'
        else:
            return 'FN'


def wsola_sample(sample, speed):
    '''Scales sample by speed'''
    sample = sample.reshape(1, len(sample))
    reader = ArrayReader(sample)
    writer = ArrayWriter(channels=1)
    tsm = wsola(1, speed=speed)
    tsm.run(reader, writer)
    return writer.data[0]


a = []  # used to calculating m and s


def classify(test, train, k=5, scale=False, verification=True, coeff=3):
    '''Classifies each observation in `test` based on `train` using the described models.
    Parameter `scale` controls whether wsola scaling should be used.
    WARNING: Parameter `k` is not used and `verification` should only be True.'''
    for s1 in test:
        test_data = s1.data
        test_mfcc = psf.mfcc(test_data, samplerate=rate, winstep=0.016,
                             winfunc=np.hamming, nfft=1200)
        results = []
        for s2 in train:
            if s1 == s2:
                # Skip identical observations, in case it is found in both test and train set
                print('Skipping', s2)
                continue
            if verification and s1.claim != s2.name:
                # In verification mode, only check the user that the speaker is claiming to be
                continue
            if scale:
                # print('scaling')
                # print('s1', s1.size, 's2', s2.size)
                scale_speed = s1.size/s2.size
                # print('Scaling by', scale_speed)
                test_data = wsola_sample(s1.data, speed=scale_speed)
                # print('After:', test_data.shape, s2.size)
                # print('Extracting features')
                test_mfcc = psf.mfcc(test_data, samplerate=s1.rate, winstep=0.016,
                                     winfunc=np.hamming, nfft=1200)
            d, *pth = dtw(test_mfcc, s2.data, dist=distance.euclidean)
            results.append((d, s2))
        results = sorted(results)[:k]
        if verification:
            prediction = results[0]
            print(s1)
            print(prediction[1])
            print(prediction[0], '\t', end='')
            a.append(prediction[0])
            outcome = not prediction[0] > (m+coeff*s)
            yield parse(s1, prediction[1], outcome)
        else:
            # Not used anymore
            score_by_speaker = defaultdict(lambda: (0, 0))
            # name: (times_found_in_neighs, sum_of_distance)
            for res in results[:k]:
                score_by_speaker[res[1].name] = (score_by_speaker[res[1].name][0]+1, score_by_speaker[res[1].name][1]+res[0])
            # get max times found in neighs
            max_neighs = score_by_speaker[sorted(dict(score_by_speaker), key=score_by_speaker.get, reverse=True)[0]][0]
            # get neigh found max times with lowest distance
            neighs_with_max = {speaker: v[1] for speaker, v in dict(score_by_speaker).items() if v[0] == max_neighs}
            return sorted(neighs_with_max, key=neighs_with_max.get)[0] == s1.name


# Get folders in folder 'train/'
folders = sorted(os.listdir('train/'))

# The database is simply a list, as there was a lot of experimentation with different parameters and features,
# which would have resulted in dropping lots of tables.
# Can be replaced with an actual database once features have been decided upon.
db = []

for folder in folders:
    path = 'train/' + folder + '/'
    # Get all wav files sorted alphabetically, except those with 'ignore' in their names
    wavs = sorted([f for f in os.listdir(path) if '.wav' in f and 'ignore' not in f])
    for w in wavs:
        new_path = path + w
        rate, sample = wav.read(new_path)
        sample = normalize(sample)

        speech_start, speech_end = 0, len(sample)
        if DETECT_ENDPOINT:
            speech_start, speech_end = endpoint_detection(sample)

        mfcc = psf.mfcc(sample[speech_start:speech_end], samplerate=rate,
                        winstep=0.016, winfunc=np.hamming, nfft=1200)
        name = folder.split('-')[0]
        word = folder.split('-')[1]
        attempt = w.replace('.wav', '')

        # For the train set, the speaker is not claiming to be anyone else
        speaker = Speaker(name, name, word, attempt, rate, speech_end-speech_start, mfcc)
        print(speaker)
        db.append(speaker)

# # to calculate m, s without endpoint detection, exits once done
# a = []
# for s1 in db:
#     tested = []
#     print(s1)
#     for s2 in db:
#         if s1.claim != s2.name:
#             continue
#         #cost_matrix, acc_cost_matrix,
#         d, *pth = dtw(s1.data, s2.data, dist=distance.euclidean)
#         #print(s1.name, s1.id, s2.name, s2.id, d)
#         tested.append((d, s2))
#     tested = sorted(tested)
#     print(tested[1][1])
#     print(tested[1][0])
#     a.append(tested[1][0])
# import statistics
# m = statistics.mean(a)
# s = statistics.stdev(a)
# print(m, s)
# import sys
# sys.exit()

# Map speaker names to the users they're claiming to be
claim = {'Marios': 'Kostas', 'Korina': 'Eugene', 'Kor': 'Andreana'}

# For train set, similar procedure to the one followed for the test set
# Get all folders in folder 'actual_test'
folders = sorted(os.listdir('actual_test/'))
test_db = []
for folder in folders:
    path = 'actual_test/' + folder + '/'
    wavs = sorted([f for f in os.listdir(path) if '.wav' in f and 'ignore' not in f])
    for w in wavs:
        new_path = path + w
        rate, sample = wav.read(new_path)
        sample = normalize(sample)

        speech_start, speech_end = 0, len(sample)
        if DETECT_ENDPOINT:
            speech_start, speech_end = endpoint_detection(sample)

        name = folder.split('-')[0]
        word = folder.split('-')[1]
        attempt = w.replace('.wav', '')
        cl = name
        # for speakers claiming to be someone else
        if name in claim:
            cl = claim[name]
        speaker = Speaker(name, cl, word, attempt, rate, speech_end-speech_start, sample)
        print('Test', speaker)
        test_db.append(speaker)

# Output to write to data file for plotting
output = '@coeff\taccuracy\tprecision\trecall\tf1_score\n'
# Repeat for coefficients in the following range divided by 2 ([-1.5, 4.5) with 0.5 step)
for co in range(-3, 9, 1):
    confusion_matrix = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    for i in classify(test_db, db, k=5, scale=DETECT_ENDPOINT, verification=True, coeff=co/2):
        print(i)
        confusion_matrix[i] += 1
    print('--------------------------------------------')
    print(confusion_matrix['TP'], confusion_matrix['FN'], sep='\t')
    print(confusion_matrix['FP'], confusion_matrix['TN'], sep='\t')
    print('--------------------------------------------')
    tp = confusion_matrix['TP']
    fn = confusion_matrix['FN']
    fp = confusion_matrix['FP']
    tn = confusion_matrix['TN']
    try:
        # Calculate accuracy, precision, recall, f1_score
        accuracy = (tp+tn)/(tp+fp+fn+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*(recall*precision)/(recall+precision)
    except ZeroDivisionError:
        # In case any of the denominators are zero, move on to the next test
        continue
    output += '\t'.join([str(i) for i in [co/2, accuracy, precision, recall, f1_score]]) + '\n'
    print(output)
# Write to data file based on Model
with open(str(DETECT_ENDPOINT).lower()+'.dat', 'w') as f:
    f.write(output)
