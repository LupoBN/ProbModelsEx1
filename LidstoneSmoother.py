from MLEstimator import MLEstimator
from Utils import *

VOCAB_SIZE = 300000


# This class is responsible for returning the estimation for words with the Lidstone smoother.
class LidstoneSmoother:
    def __init__(self, mle, lamda, vocab_size):
        self._mle = mle
        self._lamda = float(lamda)
        self._S = mle.get_S()
        self._X = vocab_size

    def get_word_prob(self, word):
        word_freq = self._mle.get_word_freq(word)
        return (float(word_freq) + self._lamda) / (self._S + self._lamda * self._X)

    def get_f(self, frequency):
        prob_for_frequency = (frequency + self._lamda) / (self._S + self._lamda * self._X)
        return self._S * prob_for_frequency



def lidstone_prob_validity_test():
    words = read_file("develop.txt", parse_no_title, " ")
    words_set = set(words)
    ml = MLEstimator(words)

    ls = LidstoneSmoother(ml, 0.1, VOCAB_SIZE)
    probs = ls.get_word_prob("THISWORDCAN'TBEONTHEWORDLISTWHATAREYOUCRAZY?") * (VOCAB_SIZE - len(words_set))
    for word in words_set:
        probs += ls.get_word_prob(word)
    assert probs - 1.0 <= 1e-5
    print "Passed lidstone probablity validity test"


# Tests for lidstone model.
def lidstone_base_test():
    words = ["my", "friend", "walked", "to", "the", "park", "and", "ate", "the", "sandwich"]
    ml = MLEstimator(words)
    ls = LidstoneSmoother(ml, 0.1, 10)
    for word in words:
        prob = ls.get_word_prob(word)
        if word != "the":
            assert 0.1 - prob <= 0.0001
        else:
            assert 0.190909091 - prob <= 0.0001
    print "Passed lidstone probability checks"
    ls = LidstoneSmoother(ml, 2.0, 300000)
    for word in words:
        prob = ls.get_word_prob(word)
        if word != "the":
            assert 0.000005 - prob <= 0.0001
        else:
            assert 0.000006667 - prob <= 0.0001
    probs = ls.get_word_prob("THISWORDCAN'TBEONTHEWORDLISTWHATAREYOUCRAZY?")
    assert probs - 0.000003333 <= 0.0001

    print "Passed lidstone probability checks"


if __name__ == "__main__":
    lidstone_base_test()
    lidstone_prob_validity_test()
