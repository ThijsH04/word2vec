# word2vec
Jetbrain internship application

## Data
weights, training data, losses and scores can soon be found in [https://drive.google.com/drive/folders/1yFB9ciApYhACdVesXQeOKaS_EJY-t2n6?usp=drive_link](https://drive.google.com/drive/folders/1yFB9ciApYhACdVesXQeOKaS_EJY-t2n6?usp=sharing)

## How to run
### training
The training code can be run using TrainModel.py. It requires 3 arguments, the data for the input_file and analogies_file can soon be found in the google drive above
--input_file (location to the training corpus)
--analogies_file (location to file with analogy pair combinations) 
--output_directory (location where the output of the model should be stored)

Parameters for training can also be adjusted in TrainModel.py.

### loading weights
to see that the model outputted valid vector representations, the weights can also be downloaded above and AnalyseData.py can be ran. The weights provided are of dimension 300 and trained on part of the google 1 billion words dataset. Eventually the average score of the analogies came to ~19%, using the same dataset as in the orignal Word2Vector paper. The python file also requires 3 arguments, all of which should be available in the google drive
--vocab_location (location of the vocabulary)
--input_weights (location of the weights for the in matrix, the actual word2vec embeddings)
--output_weights (location of the weights for the out matrix)

The code can be adapted slightly to try different word combinations.
