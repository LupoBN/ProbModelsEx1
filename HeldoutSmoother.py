from collections import Counter
from Utils import *
X_SIZE = 300000

# This class is responsible for returning the estimation for words with the Held Out smoother.
class HeldoutSmoother:
    def __init__(self, train, heldout):
        self.N_dict = {}
        self.t_dict = {}
        self.train = train
        self.heldout = heldout
        self.train_frequencies = Counter()
        self.heldout_frequencies = Counter()
        self.train_frequencies.update(train)
        self.heldout_frequencies.update(heldout)


    def get_word_prob(self, x):
        r = self.train_frequencies[x]
        t = self.get_t(r)
        N = self.get_N(r)
        return float(t) / (len(self.heldout) * N)

    def get_f(self, frequency):
        t = self.get_t(frequency)
        N = self.get_N(frequency)
        prob_for_frequency = float(t) / (len(self.heldout) * N)
        return prob_for_frequency * len(self.train)

    # calculate t_r
    def get_t(self, r):
        if r in self.t_dict:
            return self.t_dict[r]
        if r == 0:
            self.t_dict[r] = sum([self.heldout_frequencies[k] for k in self.heldout_frequencies.keys() if self.train_frequencies[k] == 0])
        else:
            self.t_dict[r] = sum([self.heldout_frequencies[k] for k in self.train_frequencies.keys() if self.train_frequencies[k] == r])
        return self.t_dict[r]

    # calculate N_r
    def get_N(self, r):
        if r in self.N_dict:
            return self.N_dict[r]
        if r == 0:
            self.N_dict[r] = X_SIZE - len(self.train_frequencies.keys())
        else:
            self.N_dict[r] = len([k for k in self.train_frequencies.keys() if self.train_frequencies[k] == r])
        return self.N_dict[r]


def heldout_prob_validity_test():
    words = read_file("develop.txt", parse_no_title, " ")
    train_percent = int(round(0.5 * len(words)))
    train_set = words[:train_percent]
    heldout_set = words[train_percent:]
    words_set = set(words)
    ml = MLEstimator(words)
    ho = HeldoutSmoother(train_set, words_set)
    probs = ho.get_word_prob("THISWORDCAN'TBEONTHEWORDLISTWHATAREYOUCRAZY?") * (X_SIZE - len(set(train_set)))
    for word in set(train_set):
        probs += ho.get_word_prob(word)
    assert probs - 1.0 <= 1e-5
    print "Passed heldout probablity validity test"


if __name__ == '__main__':
    heldout_prob_validity_test()