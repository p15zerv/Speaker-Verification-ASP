import scipy.io.wavfile as wav
import numpy as np


class Speaker:
    '''Simple data class to represent a speaker'''
    def __init__(self, name, claim, word, attempt, rate, size, data):
        self.name = name
        self.claim = claim
        self.word = word
        self.attempt = attempt
        self.rate = rate
        self.size = size
        self.data = data

    def __eq__(self, other):
        if (self.name, self.claim, self.word, self.attempt, self.size) == (other.name, self.claim, other.word, other.attempt, other.size):
            return True
        return False

    def __repr__(self):
        return f'Speaker(name={self.name}, claim={self.claim}, word={self.word}, attempt={self.attempt}, size={self.size})'


def windowing(sample, rate, win_size, overlap, window_type='time',
              overlap_type='abs'):
    '''Windowing implementation using generators so that windows are lazily calculated.'''
    if window_type == 'time':
        win_size = win_size * int(rate / 1000)
        if overlap_type == 'abs':
            overlap = overlap * int(rate/1000)
    elif window_type != 'samples':
        raise ValueError('Window type must be either "time" or "samples"')
    if overlap_type == 'percent':
        overlap = round(overlap*win_size)

    left = 0
    right = min(win_size, len(sample))
    while right < len(sample):
        yield sample[left:right]
        left += win_size - overlap
        right = min(left+win_size, len(sample))
    yield sample[left:right]


def zero_crossing_rate(segment):
    zc = 0
    for a, b in zip(segment, segment[1:]):
        if np.sign(a) * np.sign(b) <= 0:
            zc += 1
    return zc/len(segment)


def frame_pow(segment):
    fpow = np.sum(segment**2)/len(segment)
    return fpow


def frame_energy(segment):
    return np.sum(np.abs(segment))/len(segment)


def normalize(sample):
    '''Normalize signal to [-1, 1] by converting to type int32, then diving by int32's maximum'''
    sample = sample.astype('int32')/32768
    return sample


def endpoint_detection(sample, rate=48000, win_size=25, win_overlap=16):
    def detect(zcr, energy, reverse=False):
        '''Responsible for the detection.
        `reverse`=True reverses the windows to find the end.
        WARNING: Assumes 48kHz!'''
        counter = 0
        data = list(zip(zcr, energy))
        if reverse:
            data = reversed(data)

        for i, (z, e) in enumerate(data):
            # if e > e_threshold:
            if z < 5*z_threshold and e > e_threshold:
                counter += 1
                if counter == 1:
                    first = i
                # print((len(zcr)-i)*432, e)
                if counter == 7:
                    if reverse:
                        # print((len(zcr) - first) * 432)
                        return (len(zcr) - first) * 432
                    return first*432
            else:
                counter = 0
    zcr = []
    energy = []
    for window in windowing(sample, rate, 25, 16):
        zcr.append(zero_crossing_rate(window))
        energy.append(frame_energy(window))
    z_threshold = (zero_crossing_rate(sample[:100*48]) + zero_crossing_rate(sample[-100*48:]))/2
    e_threshold = frame_energy(sample[:100*48])
    start, end = detect(zcr, energy), detect(zcr, energy, reverse=True)
    if start is None:
        start = 0
    if end is None:
        end = len(sample)
    return start, end


if __name__ == '__main__':
    (rate, sample) = wav.read('joe.wav', )
    sample = sample.astype('int32')
    sample = normalize(sample)
    print(rate, sample)
    print(len(list(windowing(sample, rate, 25, 0))))
    print(zero_crossing_rate(sample))
    print(frame_pow(sample))
