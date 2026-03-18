# for the text dataset, i downloaded it from the terminal with curl -O http://mattmahoney.net/dc/text8.zip and then i unzipped it
import numpy as np

#getting the list of words
with open("text8") as file:
    words = file.read().split()
words = words[:3_000_000]
#mapping the word frequencies
word_frequency_dict = {}
for word in words:
    word_frequency_dict[word] = word_frequency_dict.get(word, 0) + 1
word_frequency_dict = dict(sorted(word_frequency_dict.items()))


#a heuristic to remove words that appear ofthen with a probability of sqrt(1e-3/frequency of the word)
freqs = {word: count/len(words) for word, count in word_frequency_dict.items()}
keep_prob = {word: min(1, np.sqrt(1e-3 / freqs[word])) for word in word_frequency_dict}
words = [w for w in words if np.random.random() < keep_prob[w]]

#word to index mapping
wordtoi = {word:i for i, word in enumerate(word_frequency_dict.keys())}
itoword = {i:word for word, i in wordtoi.items()}

#array thats the same as the words array, just with indexes instead of words
words_index_array = np.array([wordtoi[word] for word in words], dtype= np.int32)

#array of the distribution used for negative sampling, pword probability is proportional to frequency ^0.75
neg_sampling_distribution = np.array([i**0.75 for i in word_frequency_dict.values()])
neg_sampling_distribution /= np.sum(neg_sampling_distribution)

def get_pos_examples(x_positions, max_window_size):
    """
    A function that takes in a numpy list of word positions in the text and outputs a numpy list od positive examples that correspond to each position in x.
    """
    window_range= np.array(list(range(-max_window_size, 0)) + list(range(1, max_window_size+1)))
    positive_sample_positions = x_positions + np.random.choice(window_range, size = len(x_positions))
    return words_index_array[positive_sample_positions]

def get_neg_examples(batch_size, k):
    """
    returns a numpy array of size (batch_size, k) that is populeated with random indexes od words based on the frequency of the word to the raised to ^0.75 
    """
    res = np.random.multinomial(n= 1, pvals= neg_sampling_distribution, size=batch_size*k)
    res = np.argmax(res, axis=-1)
    res = res.reshape(batch_size, k)
    return res

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batch_generator(batch_size, k, max_window_size):
    """
    A generator function to get the (x, y) batch pair
    Args:
    batch_size
    k - the numer of negative samples per 1 positive sample
    max_window_size - for each target word the window size will be a random number uniformly taken from [1, max_window_size]

    Returns:
    x - numpy tensor of size (batch_size, 1) with indexes of words
    y - numpy tensor of size (batch_size, k+1) where y[:, 0] are the positive samples and y[:, 1:] are the negative ones
    """
    #to ensure sampling without replacement, and that window size doesnt get out of bounds
    indexes = np.arange(max_window_size, len(words_index_array)-max_window_size, 1)
    while True:
        np.random.shuffle(indexes)
        for i in range(0, len(indexes)-batch_size+1, batch_size):
            x_positions = indexes[i:i+batch_size]
            y = np.zeros((batch_size, k+1), dtype= np.int32)
            y[:, 0] = get_pos_examples(x_positions, max_window_size)
            y[:, 1:] = get_neg_examples(batch_size, k)
            x = words_index_array[x_positions]
            x = x[..., np.newaxis]
            yield x, y





class SkipGram:

    def __init__(self, vocab_size, emb_dim):

        #embeddings initialized with mean 0 and std of 1/sqrt(emb_dim)
        scale = 1 / np.sqrt(emb_dim)
        self.target_embedding = np.random.normal(0, scale, (vocab_size, emb_dim)).astype(np.float32)
        self.context_embedding = np.random.normal(0, scale, (vocab_size, emb_dim)).astype(np.float32)

        #gradients are 0 at the beginning
        self.dtarget_embedding = np.zeros_like(self.target_embedding)
        self.dcontext_embedding = np.zeros_like(self.context_embedding)

    def zero_grad(self):

        self.dtarget_embedding = np.zeros_like(self.target_embedding)
        self.dcontext_embedding = np.zeros_like(self.context_embedding)

    def forward(self, x, y, calculate_loss = False):
        
        target = self.target_embedding[x] # (batch_size, 1, emb_dim)
        context = self.context_embedding[y] # (batch_size, k+1, emb_dim)
        context = context.transpose(0, 2, 1) # (batch_size, emb_dim, k+1)
        logits = target @ context # (batch_size, 1, k+1)
        self.logits = logits

        if calculate_loss:
            logits_sig = sigmoid(logits)
            logits_sig[:, :, 1:] = 1 - logits_sig[:, :, 1:]
            loss = -np.log(logits_sig)
            loss = np.sum(loss)
            #dividing the loss by batch size
            loss /= logits.shape[0]
            return loss


    def backward(self, x, y):

        assert hasattr(self, "logits")

        batch_size = self.logits.shape[0]

        #the derivative with respect to logits is sigmoid(x) for neg examples and sigmoid(x) - 1 for positive examples
        dlogits = sigmoid(self.logits)
        dlogits[:, :, 0] -= 1
        dlogits /= batch_size # (batch_size, 1, k+1)

        #chain rule for matrix multiplication
        #updating only rows involved in calculation
        #in case of there being duplicates in x or y, theres no problem as the gradient is accumulated, not overwritten
        np.add.at(self.dtarget_embedding, x, dlogits @ self.context_embedding[y]) # (batch_size, 1, k+1) @ (batch_size, k+1, emb_dim)
        np.add.at(self.dcontext_embedding, y, dlogits.transpose(0, 2, 1) @ self.target_embedding[x]) # (batch_size, k+1, 1) @ (batch_size, 1, emb_dim)

    def update(self, learning_rate, weight_decay= 0):
        
        assert weight_decay >= 0 and weight_decay < 1

        self.context_embedding = self.context_embedding*(1- weight_decay) - learning_rate*self.dcontext_embedding
        self.target_embedding = self.target_embedding*(1- weight_decay) - learning_rate*self.dtarget_embedding