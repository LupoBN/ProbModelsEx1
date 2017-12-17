import sys
from Utils import *
from LidstoneSmoother import LidstoneSmoother
import math
from HeldoutSmoother import HeldoutSmoother

STUDENTS = "Matan Ben Noach Itay Mosafi 201120441 205790983\n"


VOCAB_SIZE = 300000


# Calculate the perplexity of a given estimator on the validation data.
def calculate_perplexity(estimator, validation):
    log_likelihood = 0.0
    for word in validation:
        # Estimate the probability of a word in the validation set.
        prob = estimator.get_word_prob(word)
        try:
            # Calculate the log likelihood for a word in the validation set.
            log_likelihood += math.log(prob, 2)
        except:
            log_likelihood += -float("Inf")
    # Take the negative average of the log likelihood.
    log_likelihood /= -len(validation)
    # Return 2 in power of the log likelihood to retrieve the perplexity.
    return pow(2, log_likelihood)


# find the best lamda from 0.00 to 2.00 on the validation set.
def find_best_lambda(validation, ml):
    best_lamda = 0.0
    best_lamda_perplixity = float('Inf')
    lamda = 0.00
    for i in range(0, 201):
        ls = LidstoneSmoother(ml, lamda, VOCAB_SIZE)
        current_perplixity = calculate_perplexity(ls, validation)
        if current_perplixity < best_lamda_perplixity:
            best_lamda = lamda
            best_lamda_perplixity = current_perplixity
        lamda += 0.01
    return best_lamda, best_lamda_perplixity


# separate the data to train and validation.
def separate_data(words):
    output_str = str()
    train_percent = int(round(0.9 * len(words)))
    train = words[:train_percent]
    validation = words[train_percent:]
    train_set = set(train)
    output_str += "Output8 " + str(len(validation)) + "\n"
    output_str += "Output9 " + str(len(train)) + "\n"
    output_str += "Output10 " + str(len(train_set)) + "\n"
    return train, validation, output_str


# Create the string to output one to six.
def output_one_to_six():
    output_str = str()
    output_str += "Output1 " + sys.argv[1] + "\n"
    output_str += "Output2 " + sys.argv[2] + "\n"
    output_str += "Output3 " + sys.argv[3] + "\n"
    output_str += "Output4 " + sys.argv[4] + "\n"
    output_str += "Output5 " + str(VOCAB_SIZE) + "\n"
    output_str += "Output6 " + str(1.0 / float(VOCAB_SIZE)) + "\n"
    return output_str


# Create the string to output twelve to twenty
def output_twelve_to_twenty(ml, input_word, validation, best_lamda):
    output_str = str()
    input_word_prob = ml.estimate_word(input_word)
    unknown_word_prob = ml.estimate_word("THISWORDCAN'TBEONTHEWORDLISTWHATAREYOUCRAZY?")
    output_str += "Output12 " + str(input_word_prob) + "\n"
    output_str += "Output13 " + str(unknown_word_prob) + "\n"
    ls = LidstoneSmoother(ml, 0.10, VOCAB_SIZE)
    input_word_prob = ls.get_word_prob(input_word)
    unknown_word_prob = ls.get_word_prob("THISWORDCAN'TBEONTHEWORDLISTWHATAREYOUCRAZY?")
    perplixity10 = calculate_perplexity(ls, validation)
    output_str += "Output14 " + str(input_word_prob) + "\n"
    output_str += "Output15 " + str(unknown_word_prob) + "\n"
    ls = LidstoneSmoother(ml, 0.01, VOCAB_SIZE)
    perplixity = calculate_perplexity(ls, validation)
    output_str += "Output16 " + str(perplixity) + "\n"
    output_str += "Output17 " + str(perplixity10) + "\n"
    ls = LidstoneSmoother(ml, 1.0, VOCAB_SIZE)
    perplixity = calculate_perplexity(ls, validation)
    output_str += "Output18 " + str(perplixity) + "\n"
    output_str += "Output19 " + str(best_lamda) + "\n"
    ls = LidstoneSmoother(ml, best_lamda, VOCAB_SIZE)
    perplixity = calculate_perplexity(ls, validation)
    output_str += "Output20 " + str(perplixity) + "\n"
    return output_str


def separate_data_heldout(words):
    train_percent = int(round(0.5 * len(words)))
    train_set = words[:train_percent]
    heldout_set = words[train_percent:]
    return train_set, heldout_set


def output_21_to_28(words, ml, heldout_smoother):
    output_str = str()
    ls = LidstoneSmoother(ml, 0.06, VOCAB_SIZE)

    train_set, heldout_set = separate_data_heldout(words)
    output_str += "Output21 " + str(len(train_set)) + "\n"
    output_str += "Output22 " + str(len(heldout_set)) + "\n"
    output_str += "Output23 " + str(heldout_smoother.get_word_prob(sys.argv[3])) + "\n"
    output_str += "Output24 " + str(
        heldout_smoother.get_word_prob("THISWORDCAN'TBEONTHEWORDLISTWHATAREYOUCRAZY?")) + "\n"
    words = read_file(sys.argv[2], parse_no_title, " ")
    output_str += "Output25 " + str(len(words)) + "\n"
    ls_perplexity = calculate_perplexity(ls, words)
    ho_perplexity = calculate_perplexity(heldout_smoother, words)
    better_model = 'L' if ls_perplexity < ho_perplexity else 'H'
    output_str += "Output26 " + str(ls_perplexity) + "\n"
    output_str += "Output27 " + str(ho_perplexity) + "\n"
    output_str += "Output28 " + better_model + "\n"
    return output_str


def output_table(best_lambda, ml, heldout_smoother):
    output_str = "Output29\n"
    ls = LidstoneSmoother(ml, best_lambda, VOCAB_SIZE)
    for i in xrange(10):
        output_str += str(i) + "\t" + "{0:.5f}".format(ls.get_f(i)) + '\t' + "{0:.5f}".format(
            heldout_smoother.get_f(i)) + '\t' + str(heldout_smoother.N_dict[i]) + '\t' + str(
            heldout_smoother.t_dict[i]) + '\n'

    return output_str


if __name__ == "__main__":
    output_content = STUDENTS
    output_content += output_one_to_six()
    words = read_file(sys.argv[1], parse_no_title, " ")
    ml = MLEstimator(words)
    output_content += "Output7 " + str(len(words)) + "\n"
    train, validation, output_str = separate_data(words)
    output_content += output_str
    ml = MLEstimator(train)
    num_of_input = ml.get_word_freq(sys.argv[3])
    output_content += "Output11 " + str(num_of_input) + "\n"
    best_lamda, best_perplexity = find_best_lambda(validation, ml)
    output_content += output_twelve_to_twenty(ml, sys.argv[3], validation, best_lamda)
    train_percent = int(round(0.5 * len(words)))
    train_set = words[:train_percent]
    heldout_set = words[train_percent:]
    heldout_smoother = HeldoutSmoother(train_set, heldout_set)
    output_content += output_21_to_28(words, ml, heldout_smoother)
    output_content += output_table(best_lamda, ml, heldout_smoother)
    write_file(sys.argv[4], output_content)
