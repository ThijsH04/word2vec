import numpy as np
import argparse
import matplotlib.pyplot as plt

from Word2Vec import Word2Vec


def main(args):
    """
    Overarching function for the data analysis
    :param args: file locations
    """
    model = Word2Vec(np.array(["dummy"]),10)
    model.load_from_file(args.vocab_location, args.input_weights, args.output_weights)
    training_losses = np.loadtxt(args.training_loss, delimiter=",").flatten() if args.training_loss else None
    test_losses = np.loadtxt(args.test_loss, delimiter=",").flatten() if args.test_loss else None
    recalls = np.loadtxt(args.analogy_scores, delimiter=",") if args.analogy_scores else None

    if training_losses is not None or test_losses is not None:
        plot_losses(test_losses, training_losses)

    if recalls is not None:
        plot_recalls(recalls)

    analogy_tests(model)


def analogy_tests(model):
    """
    Allow the user to input 3 words of an analogy and let the system retrieve k possibilities for the missing word
    This functions keeps looping
    :param model: Word2Vec model with loaded weights
    """
    while True:
        inputted_words = []
        word_indices = []
        word_text = ["first", "second", "third"]
        success = True
        for i in range(3):
            inputted_word = input("Enter " + word_text[i] + " word of the analogy:").lower()
            if inputted_word not in model.vocab:
                success = False
                break
            inputted_words.append(inputted_word)
            word_indices.append(np.where(model.vocab == inputted_word)[0][0])
        if not success:
            input("that word is not in the vocabulary, press enter and try again")
            continue
        k = int(input("Enter the number of words to recall:"))
        while k < 1 or k >= len(model.vocab):
            k = int(input("pick a k between 1 and" + str(len(model.vocab) - 1)) + ": ")
        recall_result = model.recall_k(word_indices, k)
        for i in range(len(recall_result)):
            print(i + 1, recall_result[i])


def plot_losses(test_losses,training_losses):
    """
    Plot the losses
    :param test_losses: np array of test losses
    :param training_losses: np array of training losses
    """
    epochs = range(1, training_losses.shape[0] + 1)
    plt.figure(figsize=(10, 5))
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, training_losses, label="Train")
    plt.plot(epochs, test_losses, label="Test")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_recalls(recalls):
    """
    Plot the recalls
    :param recalls: numpy array of recalls
    """
    plt.figure(figsize=(10, 5))
    plt.title("Average Recall@k")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@k")

    epochs = range(1, recalls.shape[0] + 1)
    for k in range(0,recalls.shape[1]):
        plt.plot(epochs, recalls[:,k], label=str(k+1))
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_location", help="location of the vocabulary file", required=True)
    parser.add_argument("--input_weights", help="location of the input weights file", required=True)
    parser.add_argument("--output_weights", help="location of the output weights file", required=True)
    parser.add_argument("--training_loss", help="location of the training loss file")
    parser.add_argument("--test_loss", help="location of the test loss file")
    parser.add_argument("--analogy_scores", help="location of the analogy scores file")
    args = parser.parse_args()
    main(args)