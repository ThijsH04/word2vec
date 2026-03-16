import numpy as np
import argparse

from Word2Vec import Word2Vec


def main(args):
    model = Word2Vec(np.array(["dummy"]),10)
    model.load_from_file(args.vocab_location, args.input_weights, args.output_weights)
    analogy = np.array([[np.where(model.vocab=="jumping")[0][0],np.where(model.vocab=="jumped")[0][0],np.where(model.vocab=="falling")[0][0],np.where(model.vocab=="fell")[0][0]]])
    print(model.calculate_analogy_score(analogy,100))
    print(model.recall_k(np.array(["jumping","jumped","falling"]),100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_location")
    parser.add_argument("--input_weights")
    parser.add_argument("--output_weights")
    args = parser.parse_args()
    main(args)