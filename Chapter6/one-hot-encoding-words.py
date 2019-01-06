import numpy as np

samples = ['The cat sat on the mat', 'The dog ate my homework']
# EXAMPLE: TOKEN The: 1
# cat: 2
# sat: 3
# on: 4
# the: 5
# mat: 6
# dog: 7
# ate: 8
# my: 9
# homework: 10

#  First, build an index of all tokens in the data
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            # Assign a unique index to each unique word
            token_index[word] = len(token_index) + 1
            print(word)


print(token_index.values())

# we vectorize our samples
max_length = 10

# results
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

print(results)
