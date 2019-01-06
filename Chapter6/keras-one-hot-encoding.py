from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat', 'The dog ate my homework']

tokenizer = Tokenizer(num_words=50)

tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

print(one_hot_results)