import os
import numpy as np
import argparse

from DataSetHandler import DataSetHandler
from Utility import parse_sentences, build_neg_probs, parse_analogies, train_loop
from Word2Vec import Word2Vec

EPOCHS = 20
K = 10
WINDOW = 5
LR = 0.0005
DIMENSION = 1
SEED = 42
RECALL_K = 5

def main(args):
    with open(args.input_file, "r", encoding="utf-8") as f:
        word_data = f.read().split("\n")
    sentence_data, vocab_data, word_to_index, token_counts = parse_sentences(word_data, 5)
    print(len(vocab_data))

    neg_probs = build_neg_probs(token_counts) # Maybe should have been done only with the train set, but not enough time to retrain

    with open(args.analogies_file, "r") as f:
        analogies_data = [analogy.split(" ") for analogy in f.read().lower().split("\n")]

    word2vec = Word2Vec(np.array(vocab_data), DIMENSION, seed=SEED)

    number_of_words = len(word_to_index)
    parsed_analogies = parse_analogies(analogies_data, word_to_index)

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    data_set_handler = DataSetHandler(sentence_data[:20], 0.8)
    np.savetxt(output_directory+"/vocab.csv", vocab_data, delimiter=",", fmt="%s", encoding="utf-8")
    training_losses = []
    test_losses = []
    all_analogy_scores = []
    for epoch in range(EPOCHS):
        lr = LR * (1 - epoch/EPOCHS)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("Training set")
        training_set = data_set_handler.get_training_set()
        train_result = train_loop(training_set, WINDOW, K, neg_probs, True, number_of_words, lr, word2vec, parsed_analogies, RECALL_K)

        word2vec.write_to_file(output_directory+"/"+str(epoch)+"_")
        training_losses.append(train_result[0])
        all_analogy_scores.append(train_result[1])

        print("Test set")
        test_set = data_set_handler.get_test_set()
        test_losses.append(train_loop(test_set, WINDOW, K, neg_probs, False, number_of_words, lr, word2vec, parsed_analogies, RECALL_K)[0])

    np.savetxt(output_directory+"/training_losses", training_losses, delimiter=",")
    np.savetxt(output_directory+"/test_losses", test_losses, delimiter=",")
    np.savetxt(output_directory+"/all_analogy_scores", all_analogy_scores, delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--analogies_file")
        parser.add_argument("--output_directory")
    args = parser.parse_args()
    main(args)

