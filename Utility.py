import string
import numpy as np

def sigmoid(dot_products):
    """
    Turn dot_products into probabilities using the sigmoid function
    :param dot_products: numpy array of dot products between vectors
    :return: numpy array of floats of probabilities result from the dot_products, following the sigmoid function
    """
    return 1/(1+np.exp(-dot_products))

def calculate_loss(probs_out, labels):
    """
    Calculate the loss based on the calculate probabilities of the words and their labels
    :param probs_out: numpy array of probabilities, where the goal is to have a probability of 1 for positive words and 0 for negative words
    :param labels: numpy array of labels, where 1 represents a positive label and 0 a negative label
    :return: float of the loss
    """
    return -np.sum(np.log(probs_out[labels==1]+10e-7))-np.sum(np.log(1-probs_out[labels==0]+10e-7))

def parse_analogies(analogies_data, word_to_index):
    """

    :param analogies_data:
    :param word_to_index:
    :return:
    """
    analogies = []
    for unparsed_analogy in analogies_data:
        if len(unparsed_analogy) != 4:
            continue
        in_vocab = True
        for word in unparsed_analogy:
            if word not in word_to_index:
                in_vocab = False
                break
        if not in_vocab:
            continue
        number_representation = []
        for i in range(4):
            number_representation.append(word_to_index[unparsed_analogy[i]])
        analogies.append(number_representation)
    return analogies


def parse_sentences(sentences, minimum_freq=3, t=10e-4, seed=42):
    np.random.seed(seed)
    pre_filter_sentences = []
    word_count = {}
    number_of_words = 0
    for sentence in sentences:
        parsed_sentence = parse_sentence(sentence)
        pre_filter_sentences.append(parsed_sentence)
        number_of_words += len(parsed_sentence)
        for word in parsed_sentence:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    filtered_sentences = []
    word_to_index = {}
    vocab = []
    token_counts = []  # track counts in vocab order for negative sampling
    for sentence in pre_filter_sentences:
        new_sentence = []
        for word in sentence:
            if word == "unk":
                continue
            if word_count[word] >= minimum_freq:
                if np.random.rand() > max(0, 1 - np.sqrt(t / (word_count[word] / number_of_words))):
                    if word not in word_to_index:
                        word_to_index[word] = len(word_to_index)
                        vocab.append(word)
                        token_counts.append(word_count[word])
                    new_sentence.append(word_to_index[word])
        if len(new_sentence) > 0:
            filtered_sentences.append(new_sentence)
    return filtered_sentences, vocab, word_to_index, np.array(token_counts)

def parse_sentence(sentence):
    """
    helper function for parse_sentences, which removes all punctuation, puts all words in lower case and then turns the
    sentence string into a list of words
    :param sentence: sentence that has to be parsed into list
    :return: a list of string of all parsed words in the sentence
    """
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    return sentence.lower().strip().split()

def build_neg_probs(token_counts):
    """
    Create probability distribution for sampling a word when performing negative sampling
    :param token_counts: numpy array count of number of occurrences of words in the data set
    :return: probability distribution for sampling each word
    """
    probs = token_counts ** 0.75
    return probs / probs.sum()

def train_loop(data_set, window, k, neg_probs, do_train, number_of_words, lr, word2vec, parsed_analogies, recall_k):
    """
    Used to perform the training steps of the training loop, performing skip gram and using negative sampling
    :param data_set: numpy array of the sentences in index form, either the training or test data
    :param window: int of the size of the context window
    :param k: int of the number of negative samples
    :param neg_probs: numpy array of the probability distribution used for negative sampling
    :param do_train: boolean for whether or not to train during this loop and calculate the analogy score
    :param number_of_words: int for the number of words
    :param lr: float for the learning rate of this training epoch
    :param word2vec: word2vec model
    :param parsed_analogies: numpy array of the analogies
    :param recall_k: number of words the recall to calculate the analogy scores
    :return: tuple of a float and an array of the analogy scores (analogy score only there if do_train)
    """
    count = 0
    total_loss = 0
    for sentence in data_set:
        for center_index in range(len(sentence)):
            window = np.random.randint(1, window+1)
            window_range = (
                list(range(max(0, center_index-window), center_index)) +
                list(range(center_index+1, min(len(sentence), center_index+window+1)))
            )
            for real_word_index in window_range:
                fake_words = np.random.choice(number_of_words, size=k, p=neg_probs)
                words = np.concatenate([[sentence[real_word_index]], fake_words])
                labels = np.array([1] + [0] * k)
                probs_out = word2vec.forward([sentence[center_index]], words)
                total_loss += calculate_loss(probs_out, labels)
                count += 1
                if do_train:
                    word2vec.backward([sentence[center_index]], words, labels, lr)
    print("loss:", total_loss/count)
    analogy_scores = []
    if do_train:
        analogy_scores = word2vec.calculate_analogy_score(parsed_analogies,recall_k)
        print("analogy score:", analogy_scores)
    return (total_loss,analogy_scores)