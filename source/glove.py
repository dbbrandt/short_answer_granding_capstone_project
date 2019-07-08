# coding: utf-8
import numpy as np

class Glove:

    # Characters that will be included in the vector of answers. Dash is from hypenated words.
    GLOVE_50_FILE = 'data/glove/glove.6B.50d.txt'
    EMBEDDING_DIM = 50

    def __init__(self, load=False, filename=GLOVE_50_FILE, embedding_dim = EMBEDDING_DIM):
        self.word2index = {}
        self.embedding_matrix = None
        self.vocab2embedding = {}
        self.embedding_dim = embedding_dim
        self.custom_embedding_matrix = None
        self.loaded = False
        if load:
            self.load_glove_embeddings(filename)

    def load_glove_embeddings(self, fp, include_empty_char=True):
        """
        Loads pre-trained word embeddings (GloVe embeddings)
            Inputs: - fp: filepath of pre-trained glove embeddings
                    - embedding_dim: dimension of each vector embedding
                    - generate_matrix: whether to generate an embedding matrix
            Outputs:
                    - word2coefs: Dictionary. Word to its corresponding coefficients
                    - word2index: Dictionary. Word to word-index
                    - embedding_matrix: Embedding matrix for Keras Embedding layer
        """
        # First, build the "word2coefs" and "word2index"
        word2coefs = {} # word to its corresponding coefficients
        word2index = {} # word to word-index
        with open(fp) as f:
            for idx, line in enumerate(f):
                try:
                    data = [x.strip().lower() for x in line.split()]
                    word = data[0]
                    coefs = np.asarray(data[1:self.embedding_dim+1], dtype='float32')
                    word2coefs[word] = coefs
                    if word not in word2index:
                        word2index[word] = len(word2index)
                except Exception as e:
                    print('Exception occurred in `load_glove_embeddings`:', e)
                    continue
            # End of for loop.
        # End of with open
        if include_empty_char:
            word2index[''] = len(word2index)
        # Second, build the "embedding_matrix"
        # Words not found in embedding index will be all-zeros. Hence, the "+1".
        vocab_size = len(word2coefs)+1 if include_empty_char else len(word2coefs)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        for word, idx in word2index.items():
            embedding_vec = word2coefs.get(word)
            if embedding_vec is not None and embedding_vec.shape[0]==self.embedding_dim:
                embedding_matrix[idx] = np.asarray(embedding_vec)
        # return word2coefs, word2index, embedding_matrix
        self.word2index = word2index
        self.embedding_matrix = np.asarray(embedding_matrix)
        self.loaded = True

    def vocab_embeddings(self, vocabulary):
        self.word2vocab = {}
        for word in vocabulary:
            if word in self.word2index:
                i = self.word2index[word]
                self.vocab2embedding[word] = self.embedding_matrix[i]
            else:
                # default to zeros if the word is not found
                self.vocab2embedding[word] = np.zeros(self.embedding_dim)
        return self.vocab2embedding

    def hash(embedding):
        return round(sum(map(abs, embedding)), 4)

    def load_custom_embedding(self, vocabulary):
        self.custom_embedding_matrix = [self.embedding_matrix[i] for i in vocabulary.keys()]

# vocabulary = "High risk problems are address in the prototype program to make sure that the program is feasible. A prototype may also be used to show a company that the software can be possibly programmed".lower().split()
# print(vocabulary)
# glove = Glove(True)
# word_embeddings = glove.vocab_embeddings(vocabulary)
# print(word_embeddings)
