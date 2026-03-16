import os
import numpy as np
import argparse

from DataSetHandler import DataSetHandler
from Utility import parse_sentences, build_neg_probs, parse_analogies, train_loop
from Word2Vec import Word2Vec

def main(args):
    """
    Handles the input data parsing, training and the saving of the word2vec model
    :param args: contains file locations, the output location and training parameters
    """
    with open(args.input_file, "r", encoding="utf-8") as f:
        word_data = f.read().split("\n")
    sentence_data, vocab_data, word_to_index, token_counts = parse_sentences(word_data, 5)
    print(len(vocab_data))

    neg_probs = build_neg_probs(token_counts) # Maybe should have been done only with the train set, but not enough time to retrain

    with open(args.analogies_file, "r") as f:
        analogies_data = [analogy.split(" ") for analogy in f.read().lower().split("\n")]

    word2vec = Word2Vec(np.array(vocab_data), args.dimension, seed=args.seed)

    number_of_words = len(word_to_index)
    parsed_analogies = parse_analogies(analogies_data, word_to_index)

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    data_set_handler = DataSetHandler(sentence_data, 0.8)
    np.savetxt(output_directory+"/vocab.csv", vocab_data, delimiter=",", fmt="%s", encoding="utf-8")
    training_losses = []
    test_losses = []
    all_analogy_scores = []
    for epoch in range(args.epochs):
        lr = args.lr * (1 - epoch/args.epochs)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("Training set")
        training_set = data_set_handler.get_training_set()
        train_result = train_loop(training_set, args.window, args.k, neg_probs, True, number_of_words, lr, word2vec, parsed_analogies, args.recall_k)

        word2vec.write_to_file(output_directory+"/"+str(epoch)+"_")
        training_losses.append(train_result[0])
        all_analogy_scores.append(train_result[1])

        print("Test set")
        test_set = data_set_handler.get_test_set()
        test_losses.append(train_loop(test_set, args.window, args.k, neg_probs, False, number_of_words, lr, word2vec, parsed_analogies, args.recall_k)[0])

        np.savetxt(output_directory+"/training_losses", training_losses, delimiter=",")
        np.savetxt(output_directory+"/test_losses", test_losses, delimiter=",")
        np.savetxt(output_directory+"/all_analogy_scores", all_analogy_scores, delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input file", required=True)
    parser.add_argument("--analogies_file", type=str, help="Analogy file", required=True)
    parser.add_argument("--output_directory", type=str, help="Output directory", required=True)

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--k", type=int, default=10, help="Number of negative samples per positive")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--dimension", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--recall_k", type=int, default=5, help="K for analogy recall@K")
    args = parser.parse_args()
    main(args)