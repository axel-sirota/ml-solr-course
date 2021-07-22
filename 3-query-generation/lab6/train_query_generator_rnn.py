#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(42)
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import os
import time
import sys


# In[2]:


# In case your sys.path does not contain the base repo, go there.
print(sys.path)
get_ipython().run_line_magic('cd', "'/Users/axelsirota/repos/ml-solr-course'")


# In[3]:


path = "dataset/train_corpus_descriptions_airbnb.csv"
# Read, then decode for py2 compat.
text = open(path, 'rb').read().decode(encoding='utf-8')[:10000000]
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')


# In[4]:


# Take a look at the first 250 characters in text
print(text[:250])


# In[5]:


vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')


# In[6]:


ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)


# In[7]:


chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


# In[8]:


def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


# In[9]:


ids = ids_from_chars(tf.strings.unicode_split('Only you can prevent forest fires', input_encoding='UTF-8'))
ids


# In[10]:


text_from_ids(ids)


# In[ ]:


#Prepare the dataset

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 75  # Maximum alternate query size
examples_per_epoch = len(text)//(seq_length+1)
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


# In[ ]:


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# In[ ]:


split_input_target(list("Apache Solr"))


# In[ ]:


dataset = sequences.map(split_input_target)


# In[ ]:


for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())


# In[ ]:


# Batch size
BATCH_SIZE = 64
EPOCHS = 10
BUFFER_SIZE = 10000
vocab_size = len(vocab)
embedding_dim = 100
rnn_units = 1024

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
)


# In[ ]:


class QueryGenerator(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.rnn = tf.keras.layers.GRU(rnn_units,
                                         activation='relu',
                                         return_sequences=True,
                                         return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.rnn.get_initial_state(x)
    x, states = self.rnn(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


# In[ ]:


model = QueryGenerator(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)


# In[ ]:


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))


# In[ ]:


# Directory where the checkpoints will be saved
checkpoint_dir = './3-query-generation/lab6/.training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# In[ ]:


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# In[ ]:


class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states


# In[ ]:


one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
start = time.time()
states = None
next_char = tf.constant(['Midtown Sunny 2-Bedroom'])
result = [next_char]

for n in range(50):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result, '\n\n' + '_'*80)
print('\nRun time:', end - start)


# In[ ]:


tf.saved_model.save(one_step_model, '3-query-generation/lab6/alternative_queries')

