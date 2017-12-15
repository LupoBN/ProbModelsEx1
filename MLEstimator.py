from collections import Counter

# The MLEstimator class takes a list of words and can return the probability for a unigram or a bigram.
class MLEstimator:
    # Constructor for MLEstimator, takes a list of words and builds the frequency and probabilities
    # for each word (works for unigram and bigram).
    def __init__(self, words):
        self._frequencies = Counter()
        self._frequencies.update(words)
        self._S = float((sum(self._frequencies.values())))

    def unique_words_count(self):
        return self._words_count

    # Gets a list of words and a parameter n and returns a list of ngrams words according to the words list.
    def _create_ngrams(self, words, n):
        ngram_words = list()
        for i, word in enumerate(words[:len(words) - n + 1]):
            ngram = word
            for j in range(i + 1, i + n):
                ngram += " " + words[j]
            ngram_words.append(ngram)
        return ngram_words

    # Returns the frequency of a unigram or bigram word.
    def get_word_freq(self, word):
        return self._frequencies[word]

    # Return number of words in the list given.
    def get_S(self):
        return self._S

    # Returns the estimation for the probability of a unigram or a bigram word.
    def estimate_word(self, word):
        words = word.split(" ")
        if len(words) == 2:
            return float(self._frequencies[word]) / self._frequencies[word]
        return float(self._frequencies[word]) / self._S

# Tests for unigram model.
def unigram_test():
    words = ["my", "friend", "walked", "to", "the", "park", "and", "ate", "the", "sandwich"]
    ml = MLEstimator(words)
    for word in words:
        freq = ml.get_word_freq(word)
        if word != "the":
            assert freq == 1
        else:
            assert freq == 2
    print "Passed unigram frequency checks"
    for word in words:
        prob = ml.estimate_word(word)
        if word != "the":
            assert (1.0 / float(len(words))) - prob <= 0.0001
        else:
            assert (2.0 / float(len(words))) - prob <= 0.0001
    print "passed unigram estimates checks"

# Tests for bigram model.
def bigram_test():
    words = ["a", "friend", "walked", "up", "to", "a", "friend"]
    ml = MLEstimator(words)
    for i in range(len(words) - 1):
        word = words[i] + " " + words[i + 1]
        freq = ml.get_word_freq(word)
        if word != "a friend":
            assert freq == 1
        else:
            assert freq == 2
    print "Passed bigram frequency checks"
    for i in range(len(words) - 1):
        word = words[i] + " " + words[i + 1]

        prob = ml.estimate_word(word)
        if word != "a friend":
            assert (1.0 / ml.get_word_freq(words[i])) - prob <= 0.0001
        else:
            assert (2.0 / ml.get_word_freq(words[i])) - prob <= 0.0001
    print "passed bigram estimates checks"

if __name__ == '__main__':
    unigram_test()
