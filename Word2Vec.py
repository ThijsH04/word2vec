import numpy as np

from Utility import sigmoid


class Word2Vec:
    def __init__(self, vocab, dimension, seed=42):
        """
        Initialise Word2Vec object that uses skip-gram to train the weight vectors.
        Eventual vector representations are stored in self.w_in
        :param vocab: list of words that the dataset is made up off
        :param dimension: dimension of the vector representations
        :param seed: seed used to reproduce the Xavier initialisation of the weights
        """
        self.vocab = vocab
        self.dimension = dimension
        self.seed = seed
        np.random.seed(self.seed)

        self.w_in = np.random.normal(0, np.sqrt(2/(2*vocab.shape[0])), (vocab.shape[0], dimension))
        self.w_out = np.random.normal(0, np.sqrt(2/(2*vocab.shape[0])), (dimension, vocab.shape[0]))
        self.m_in = np.zeros_like(self.w_in)
        self.v_in = np.zeros_like(self.w_in)
        self.m_out = np.zeros_like(self.w_out)
        self.v_out = np.zeros_like(self.w_out)
        self.t = 0



    def forward(self, center, words):
        """
        Calculate the probability of the words being part of the window around the skip-gram
        :param center: the center word of the skip-gram
        :param words: possible words in the window around the center word
        :return: the probability of the words being part of the window around the center
        """
        embedding = self.w_in[center]
        values = embedding @ self.w_out[:,words]
        return sigmoid(values.squeeze())

    def backward(self, center, words, labels, learning_rate, b1=0.9, b2=0.999, e=1e-6):
        """
        Update the model parameters based on training data made up of a center word of a skip-gram and an array of
        words to trian on, with the goal of making the probability of words with label==1 equal to 1 and those with label==0
        to have a probability of 0. The parameters are updated following the approach of the Adam optimizer.
        :param center: the center word (index form) of the skip-gram
        :param words: array of possible words (index form) in the window around the center word, in general only the first word is part of the window
        :param labels: array of labels representing whether a word is in the window or not
        :param learning_rate: rate at which the parameters are updated
        :param b1: momentum parameter adam
        :param b2: velocity parameter adam
        :param e: epsilon parameter adam
        """
        embedding = self.w_in[center]
        out_vectors = self.w_out[:, words].copy()
        dot_products = (embedding @ out_vectors).squeeze()
        probabilities = sigmoid(dot_products)
        error = np.array(probabilities - labels)
        grad_out = np.outer(embedding, error)
        grad_in = out_vectors @ error
        self.t += 1

        self.m_out[:, words] = b1 * self.m_out[:, words] + (1 - b1) * grad_out
        self.v_out[:, words] = b2 * self.v_out[:, words] + (1 - b2) * grad_out ** 2
        m_out = self.m_out[:, words] / (1 - b1 ** self.t)
        v_out = self.v_out[:, words] / (1 - b2 ** self.t)
        self.w_out[:, words] -= learning_rate * m_out / (np.sqrt(v_out) + e)

        self.m_in[center] = b1 * self.m_in[center] + (1 - b1) * grad_in
        self.v_in[center] = b2 * self.v_in[center] + (1 - b2) * grad_in ** 2
        m_in = self.m_in[center] / (1 - b1 ** self.t)
        v_in = self.v_in[center] / (1 - b2 ** self.t)
        self.w_in[center] -= learning_rate * m_in / (np.sqrt(v_in) + e)

    def recall_k(self, word_indices, recall_k):
        """
        recall the closest k words of which the vector representation is the closest to
        words[1]-words[0]+words[2]'s vector representation, calculated based on cosine similarity
        :param word_indices: list of length 3 with indices of words in the vocabulary and make up
                             an analogy pair, missing the 4th word, which the model has to retrieve
        :param recall_k: number of words to recall
        :return: list of k most likely missing words in the pair of analogy pairs according to the model
        """
        vectors = self.w_in
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        result = []
        target = normalized[word_indices[1]] - normalized[word_indices[0]] + normalized[word_indices[2]]
        similarities = normalized @ target
        similarities[word_indices[0]] = -np.inf
        similarities[word_indices[1]] = -np.inf
        similarities[word_indices[2]] = -np.inf
        for i in range(recall_k):
            best_index = np.argmax(similarities)
            result.append(self.vocab[best_index])
            similarities[best_index] = -np.inf
        return result



    def calculate_analogy_score(self, analogy_combinations, recall_k):
        """
        Calculate the average recall scores of the model for different combinations of analogy pairs.
        For a combination of Pairs (a, b) and (c, d), the system should aim to produce the vector representation
        of d, when applying c + (b-a). We then recall the k vectors closest to c + (b-a) and calculate AvgRecall@k.
        :param analogy_combinations: list of analogy combinations, comprised of 2 pairs
        :param recall_k: int for up to which k the AvgRecall score should be calculated
        :return: List of AvgRecall@k scores for 1<=k<=recall_k
        """
        vectors = self.w_in
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        correct = np.zeros(recall_k)
        for ac in analogy_combinations:
            target = normalized[ac[1]] - normalized[ac[0]] + normalized[ac[2]]
            similarities = normalized @ target
            similarities[ac[0]] = -np.inf
            similarities[ac[1]] = -np.inf
            similarities[ac[2]] = -np.inf

            for i in range(len(correct)):
                most_similar = np.argmax(similarities)
                if most_similar == ac[3]:
                    correct[i:] += 1
                    break
                similarities[most_similar] = -np.inf

        return correct / len(analogy_combinations)

    def write_to_file(self, prefix):
        """
        Stores weights of model
        :param prefix: location where to store file
        """
        np.savetxt(prefix+"w_in.csv", self.w_in, delimiter=",")
        np.savetxt(prefix+"w_out.csv", self.w_out, delimiter=",")

    def load_from_file(self, vocab_location, in_location, out_location):
        """
        Load the weights of model
        :param vocab_location: file location of vocabulary
        :param in_location: file location of input weights
        :param out_location: file location of output weights
        """
        self.vocab = np.loadtxt(vocab_location, delimiter=",", dtype=str, encoding="utf-8")
        self.w_in = np.loadtxt(in_location, delimiter=",")
        self.w_out = np.loadtxt(out_location, delimiter=",")


