
## Sequential Text Generator
### Joby John (05/01/2017)
This project was completed as part of Udacity Deep Learning course. Here a RNN is trained to go over a large body of text and produce new text based on the patterns seen in the input text. For this project, the Russian novel, Anna Karennina, by Leo Tolstoy is used. 

The input: A series of text inputs A[:n-1]. The output is A[1:n]. 
Thus providing A[N] to the network will produce A[N+1] as the output, A[N+1] will produce A[N+2] and so on, allowing us to produce completely new text.

Go to the end of the notebook to see the output of the RNN to see the artificial writer's output for the sequel to Anna Karenina. Notice, how the RNN has learnt to spell words correctly, close quotation marks, end a W word (What, When, Why) with a question mark "?". 

In this notebook, I'll build a character-wise RNN trained on Anna Karenina, one of my all-time favorite books. It'll be able to generate new text based on the text from the book.

This network is based off of Andrej Karpathy's [post on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [implementation in Torch](https://github.com/karpathy/char-rnn). Also, some information [here at r2rt](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html) and from [Sherjil Ozair](https://github.com/sherjilozair/char-rnn-tensorflow) on GitHub. Below is the general architecture of the character-wise RNN.

<img src="assets/charseq.jpeg" width="500">


```python
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
```

First we'll load the text file and convert it into integers for our network to use. Here I'm creating a couple dictionaries to convert the characters to and from integers. Encoding the characters as integers makes it easier to use as input in the network.


```python
with open('anna.txt', 'r') as f:
    text=f.read()
vocab = set(text) # Extract unique leters from the body of text
vocab_to_int = {c: i for i, c in enumerate(vocab)} # Assign a number to each character.
                                                   # and create "char to int" map.
int_to_vocab = dict(enumerate(vocab)) # integer to character map
chars = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
```


```python
len(vocab) #
```




    83




```python
len(chars)
```




    1985223



Let's check out the first 100 characters, make sure everything is peachy. According to the [American Book Review](http://americanbookreview.org/100bestlines.asp), this is the 6th best first line of a book ever.


```python
text[:100]
```




    'Chapter 1\n\n\nHappy families are all alike; every unhappy family is unhappy in its own\nway.\n\nEverythin'



And we can see the characters encoded as integers.


```python
chars[:100]
```




    array([44, 57, 33, 80, 54, 38, 50, 19, 35, 63, 63, 63,  9, 33, 80, 80, 47,
           19, 26, 33, 60, 78, 18, 78, 38, 11, 19, 33, 50, 38, 19, 33, 18, 18,
           19, 33, 18, 78, 14, 38,  4, 19, 38, 32, 38, 50, 47, 19, 62, 76, 57,
           33, 80, 80, 47, 19, 26, 33, 60, 78, 18, 47, 19, 78, 11, 19, 62, 76,
           57, 33, 80, 80, 47, 19, 78, 76, 19, 78, 54, 11, 19, 67, 64, 76, 63,
           64, 33, 47, 40, 63, 63, 77, 32, 38, 50, 47, 54, 57, 78, 76], dtype=int32)



Since the network is working with individual characters, it's similar to a classification problem in which we are trying to predict the next character from the previous text.  Here's how many 'classes' our network has to pick from.


```python
np.max(chars)+1
```




    83



## Making training and validation batches

Now I need to split up the data into batches, and into training and validation sets. I should be making a test set here, but I'm not going to worry about that. My test will be if the network can generate new text.

Here I'll make both input and target arrays. The targets are the same as the inputs, except shifted one character over. I'll also drop the last bit of data so that I'll only have completely full batches.

The idea here is to make a 2D matrix where the number of rows is equal to the batch size. Each row will be one long concatenated string from the character data. We'll split this data into a training set and validation set using the `split_frac` keyword. This will keep 90% of the batches in the training set, the other 10% in the validation set.


```python
def split_data(chars, batch_size, num_steps, split_frac=0.9):
    """ 
    Split character data into training and validation sets, inputs and targets for each set.
    
    Arguments
    ---------
    chars: character array
    batch_size: Size of examples in each of batch
    num_steps: Number of sequence steps to keep in the input and pass to the network
    split_frac: Fraction of batches to keep in the training set
    
    
    Returns train_x, train_y, val_x, val_y
    """
    
    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)
    
    # Drop the last few characters to make only full batches
    x = chars[: n_batches*slice_size]
    y = chars[1: n_batches*slice_size + 1]
    
    # Split the data into batch_size slices, then stack them into a 2D matrix 
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))
    
    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps
    
    # Split into training and validation sets, keep the first split_frac batches for training
    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    
    return train_x, train_y, val_x, val_y
```

Now I'll make my data sets and we can check out what's going on here. Here I'm going to use a batch size of 10 and 50 sequence steps.


```python
train_x, train_y, val_x, val_y = split_data(chars, 10, 50)
```


```python
train_x.shape
```




    (10, 178650)



Looking at the size of this array, we see that we have rows equal to the batch size. When we want to get a batch out of here, we can grab a subset of this array that contains all the rows but has a width equal to the number of steps in the sequence. The first batch looks like this:


```python
train_x[:,:50]
```




    array([[80, 32, 42, 24, 43, 29, 74, 14, 79, 75, 75, 75, 11, 42, 24, 24, 48,
            14, 15, 42, 37,  7, 28,  7, 29, 16, 14, 42, 74, 29, 14, 42, 28, 28,
            14, 42, 28,  7, 22, 29,  2, 14, 29, 76, 29, 74, 48, 14, 47,  5],
           [14, 42, 37, 14,  5, 61, 43, 14, 49, 61,  7,  5, 49, 14, 43, 61, 14,
            16, 43, 42, 48,  6, 20, 14, 42,  5, 16, 82, 29, 74, 29, 25, 14, 21,
             5,  5, 42,  6, 14, 16, 37,  7, 28,  7,  5, 49,  6, 14, 66, 47],
           [76,  7,  5, 71, 75, 75, 20, 46, 29, 16,  6, 14,  7, 43, 67, 16, 14,
            16, 29, 43, 43, 28, 29, 25, 71, 14, 64, 32, 29, 14, 24, 74,  7, 73,
            29, 14,  7, 16, 14, 37, 42, 49,  5,  7, 15,  7, 73, 29,  5, 43],
           [ 5, 14, 25, 47, 74,  7,  5, 49, 14, 32,  7, 16, 14, 73, 61,  5, 76,
            29, 74, 16, 42, 43,  7, 61,  5, 14, 82,  7, 43, 32, 14, 32,  7, 16,
            75, 66, 74, 61, 43, 32, 29, 74, 14, 82, 42, 16, 14, 43, 32,  7],
           [14,  7, 43, 14,  7, 16,  6, 14, 16,  7, 74, 17, 20, 14, 16, 42,  7,
            25, 14, 43, 32, 29, 14, 61, 28, 25, 14, 37, 42,  5,  6, 14, 49, 29,
            43, 43,  7,  5, 49, 14, 47, 24,  6, 14, 42,  5, 25, 75, 73, 74],
           [14,  1, 43, 14, 82, 42, 16, 75, 61,  5, 28, 48, 14, 82, 32, 29,  5,
            14, 43, 32, 29, 14, 16, 42, 37, 29, 14, 29, 76, 29,  5,  7,  5, 49,
            14, 32, 29, 14, 73, 42, 37, 29, 14, 43, 61, 14, 43, 32, 29,  7],
           [32, 29,  5, 14, 73, 61, 37, 29, 14, 15, 61, 74, 14, 37, 29,  6, 20,
            14, 16, 32, 29, 14, 16, 42,  7, 25,  6, 14, 42,  5, 25, 14, 82, 29,
             5, 43, 14, 66, 42, 73, 22, 14,  7,  5, 43, 61, 14, 43, 32, 29],
           [ 2, 14, 66, 47, 43, 14,  5, 61, 82, 14, 16, 32, 29, 14, 82, 61, 47,
            28, 25, 14, 74, 29, 42, 25,  7, 28, 48, 14, 32, 42, 76, 29, 14, 16,
            42, 73, 74,  7, 15,  7, 73, 29, 25,  6, 14,  5, 61, 43, 14, 37],
           [43, 14,  7, 16,  5, 67, 43, 71, 14, 64, 32, 29, 48, 67, 74, 29, 14,
            24, 74, 61, 24, 74,  7, 29, 43, 61, 74, 16, 14, 61, 15, 14, 42, 14,
            16, 61, 74, 43,  6, 75, 66, 47, 43, 14, 82, 29, 67, 74, 29, 14],
           [14, 16, 42,  7, 25, 14, 43, 61, 14, 32, 29, 74, 16, 29, 28, 15,  6,
            14, 42,  5, 25, 14, 66, 29, 49, 42,  5, 14, 42, 49, 42,  7,  5, 14,
            15, 74, 61, 37, 14, 43, 32, 29, 14, 66, 29, 49,  7,  5,  5,  7]], dtype=int32)



I'll write another function to grab batches out of the arrays made by `split_data`. Here each batch will be a sliding window on these arrays with size `batch_size X num_steps`. For example, if we want our network to train on a sequence of 100 characters, `num_steps = 100`. For the next batch, we'll shift this window the next sequence of `num_steps` characters. In this way we can feed batches to the network and the cell states will continue through on each batch.


```python
def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape # shape of the training input train_x
    # arrs = [train_x,train_y]
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]
```

## Building the model

Below is a function where I build the graph for the network.


```python
def build_rnn(num_classes, batch_size=50, num_steps=50, lstm_size=128, num_layers=2,
              learning_rate=0.001, grad_clip=5, sampling=False):
    
    # When we're using this network for sampling later, we'll be passing in
    # one character at a time, so providing an option for that
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()
    
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # One-hot encoding the input and target characters
    x_one_hot = tf.one_hot(inputs, num_classes)
    y_one_hot = tf.one_hot(targets, num_classes)

    ### Build the RNN layers
    # Use a basic LSTM cellb
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    ### Run the data through the RNN layers
    # This makes a list where each element is on step in the sequence
    rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    
    # Run each sequence step through the RNN and collect the outputs
    outputs, state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=initial_state)
    final_state = state
    
    # Reshape output so it's a bunch of rows, one output row for each step for each batch
    seq_output = tf.concat(outputs, axis=1)
    output = tf.reshape(seq_output, [-1, lstm_size])
    
    # Now connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and batch
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    preds = tf.nn.softmax(logits, name='predictions')
    
    # Reshape the targets to match the logits
    y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    cost = tf.reduce_mean(loss)

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    # Export the nodes
    # NOTE: I'm using a namedtuple here because I think they are cool
    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                    'keep_prob', 'cost', 'preds', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph
```

## Hyperparameters

Here I'm defining the hyperparameters for the network. 

* `batch_size` - Number of sequences running through the network in one pass.
* `num_steps` - Number of characters in the sequence the network is trained on. Larger is better typically, the network will learn more long range dependencies. But it takes longer to train. 100 is typically a good number here.
* `lstm_size` - The number of units in the hidden layers.
* `num_layers` - Number of hidden LSTM layers to use
* `learning_rate` - Learning rate for training
* `keep_prob` - The dropout keep probability when training. If you're network is overfitting, try decreasing this.

Here's some good advice from Andrej Karpathy on training the network. I'm going to write it in here for your benefit, but also link to [where it originally came from](https://github.com/karpathy/char-rnn#tips-and-tricks).

> ## Tips and Tricks

>### Monitoring Validation Loss vs. Training Loss
>If you're somewhat new to Machine Learning or Neural Networks it can take a bit of expertise to get good models. The most important quantity to keep track of is the difference between your training loss (printed during training) and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). In particular:

> - If your training loss is much lower than validation loss then this means the network might be **overfitting**. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
> - If your training/validation loss are about equal then your model is **underfitting**. Increase the size of your model (either number of layers or the raw number of neurons per layer)

> ### Approximate number of parameters

> The two most important parameters that control the model are `lstm_size` and `num_layers`. I would advise that you always use `num_layers` of either 2/3. The `lstm_size` can be adjusted based on how much data you have. The two important quantities to keep track of here are:

> - The number of parameters in your model. This is printed when you start training.
> - The size of your dataset. 1MB file is approximately 1 million characters.

>These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

> - I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make `lstm_size` larger.
> - I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that helps the validation loss.

> ### Best models strategy

>The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.

>It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.

>By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.



```python
batch_size = 100
num_steps = 100 
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5
```

## Training

Time for training which is pretty straightforward. Here I pass in some data, and get an LSTM state back. Then I pass that state back in to the network so the next batch can continue the state from the previous batch. And every so often (set by `save_every_n`) I calculate the validation loss and save a checkpoint.

Here I'm saving checkpoints with the format

`i{iteration number}_l{# hidden layer units}_v{validation loss}.ckpt`


```python
epochs = 20
# Save every N iterations
save_every_n = 200
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

model = build_rnn(len(vocab), 
                  batch_size=batch_size,
                  num_steps=num_steps,
                  learning_rate=learning_rate,
                  lstm_size=lstm_size,
                  num_layers=num_layers)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Use the line below to load a checkpoint and resume training
    #saver.restore(sess, 'checkpoints/______.ckpt')
    
    n_batches = int(train_x.shape[1]/num_steps)
    iterations = n_batches * epochs
    for e in range(epochs):
        
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
            iteration = e*n_batches + b
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer], 
                                                 feed_dict=feed)
            loss += batch_loss
            end = time.time()
            print('Epoch {}/{} '.format(e+1, epochs),
                  'Iteration {}/{}'.format(iteration, iterations),
                  'Training loss: {:.4f}'.format(loss/b),
                  '{:.4f} sec/batch'.format((end-start)))
        
            
            if (iteration%save_every_n == 0) or (iteration == iterations):
                # Check performance, notice dropout has been set to 1
                val_loss = []
                new_state = sess.run(model.initial_state)
                for x, y in get_batch([val_x, val_y], num_steps):
                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    batch_loss, new_state = sess.run([model.cost, model.final_state], feed_dict=feed)
                    val_loss.append(batch_loss)

                print('Validation loss:', np.mean(val_loss),
                      'Saving checkpoint!')
                saver.save(sess, "checkpoints/i{}_l{}_v{:.3f}.ckpt".format(iteration, lstm_size, np.mean(val_loss)))
```

    Epoch 19/20  Iteration 3382/3560 Training loss: 1.1977 3.2117 sec/batch
    Epoch 20/20  Iteration 3383/3560 Training loss: 1.3074 3.2554 sec/batch
    Epoch 20/20  Iteration 3384/3560 Training loss: 1.2632 3.1774 sec/batch
    Epoch 20/20  Iteration 3385/3560 Training loss: 1.2449 3.2523 sec/batch
    Epoch 20/20  Iteration 3386/3560 Training loss: 1.2364 3.2117 sec/batch
    Epoch 20/20  Iteration 3387/3560 Training loss: 1.2242 3.2462 sec/batch
    Epoch 20/20  Iteration 3388/3560 Training loss: 1.2116 3.3439 sec/batch
    Epoch 20/20  Iteration 3389/3560 Training loss: 1.2104 3.1800 sec/batch
    Epoch 20/20  Iteration 3390/3560 Training loss: 1.2075 3.2744 sec/batch
    Epoch 20/20  Iteration 3391/3560 Training loss: 1.2069 3.2156 sec/batch
    Epoch 20/20  Iteration 3392/3560 Training loss: 1.2060 3.2246 sec/batch
    Epoch 20/20  Iteration 3393/3560 Training loss: 1.2031 3.3378 sec/batch
    Epoch 20/20  Iteration 3394/3560 Training loss: 1.2033 3.5006 sec/batch
    Epoch 20/20  Iteration 3395/3560 Training loss: 1.2037 3.3384 sec/batch
    Epoch 20/20  Iteration 3396/3560 Training loss: 1.2043 3.2775 sec/batch
    Epoch 20/20  Iteration 3397/3560 Training loss: 1.2031 3.2103 sec/batch
    Epoch 20/20  Iteration 3398/3560 Training loss: 1.2012 3.2129 sec/batch
    Epoch 20/20  Iteration 3399/3560 Training loss: 1.2010 3.3308 sec/batch
    Epoch 20/20  Iteration 3400/3560 Training loss: 1.2020 3.5401 sec/batch
    Validation loss: 1.13137 Saving checkpoint!
    Epoch 20/20  Iteration 3401/3560 Training loss: 1.2109 3.4087 sec/batch
    Epoch 20/20  Iteration 3402/3560 Training loss: 1.2121 3.3881 sec/batch
    Epoch 20/20  Iteration 3403/3560 Training loss: 1.2113 3.3215 sec/batch
    Epoch 20/20  Iteration 3404/3560 Training loss: 1.2109 3.2661 sec/batch
    Epoch 20/20  Iteration 3405/3560 Training loss: 1.2098 3.2285 sec/batch
    Epoch 20/20  Iteration 3406/3560 Training loss: 1.2095 3.2610 sec/batch
    Epoch 20/20  Iteration 3407/3560 Training loss: 1.2091 3.2024 sec/batch
    Epoch 20/20  Iteration 3408/3560 Training loss: 1.2075 3.2300 sec/batch
    Epoch 20/20  Iteration 3409/3560 Training loss: 1.2060 3.2503 sec/batch
    Epoch 20/20  Iteration 3410/3560 Training loss: 1.2062 3.1895 sec/batch
    Epoch 20/20  Iteration 3411/3560 Training loss: 1.2061 3.3341 sec/batch
    Epoch 20/20  Iteration 3412/3560 Training loss: 1.2061 3.2051 sec/batch
    Epoch 20/20  Iteration 3413/3560 Training loss: 1.2050 3.2256 sec/batch
    Epoch 20/20  Iteration 3414/3560 Training loss: 1.2038 3.3625 sec/batch
    Epoch 20/20  Iteration 3415/3560 Training loss: 1.2038 3.4671 sec/batch
    Epoch 20/20  Iteration 3416/3560 Training loss: 1.2037 3.3835 sec/batch
    Epoch 20/20  Iteration 3417/3560 Training loss: 1.2032 3.2763 sec/batch
    Epoch 20/20  Iteration 3418/3560 Training loss: 1.2029 3.2316 sec/batch
    Epoch 20/20  Iteration 3419/3560 Training loss: 1.2020 3.1658 sec/batch
    Epoch 20/20  Iteration 3420/3560 Training loss: 1.2004 3.1382 sec/batch
    Epoch 20/20  Iteration 3421/3560 Training loss: 1.1992 3.1967 sec/batch
    Epoch 20/20  Iteration 3422/3560 Training loss: 1.1990 3.3558 sec/batch
    Epoch 20/20  Iteration 3423/3560 Training loss: 1.1984 3.2704 sec/batch
    Epoch 20/20  Iteration 3424/3560 Training loss: 1.1990 3.2113 sec/batch
    Epoch 20/20  Iteration 3425/3560 Training loss: 1.1986 3.2160 sec/batch
    Epoch 20/20  Iteration 3426/3560 Training loss: 1.1979 3.2424 sec/batch
    Epoch 20/20  Iteration 3427/3560 Training loss: 1.1978 3.1910 sec/batch
    Epoch 20/20  Iteration 3428/3560 Training loss: 1.1970 3.2639 sec/batch
    Epoch 20/20  Iteration 3429/3560 Training loss: 1.1964 3.2137 sec/batch
    Epoch 20/20  Iteration 3430/3560 Training loss: 1.1960 3.2337 sec/batch
    Epoch 20/20  Iteration 3431/3560 Training loss: 1.1958 3.2439 sec/batch
    Epoch 20/20  Iteration 3432/3560 Training loss: 1.1960 3.2595 sec/batch
    Epoch 20/20  Iteration 3433/3560 Training loss: 1.1953 3.1136 sec/batch
    Epoch 20/20  Iteration 3434/3560 Training loss: 1.1959 3.1687 sec/batch
    Epoch 20/20  Iteration 3435/3560 Training loss: 1.1958 3.2551 sec/batch
    Epoch 20/20  Iteration 3436/3560 Training loss: 1.1957 3.3237 sec/batch
    Epoch 20/20  Iteration 3437/3560 Training loss: 1.1954 3.2697 sec/batch
    Epoch 20/20  Iteration 3438/3560 Training loss: 1.1953 3.3559 sec/batch
    Epoch 20/20  Iteration 3439/3560 Training loss: 1.1955 3.2862 sec/batch
    Epoch 20/20  Iteration 3440/3560 Training loss: 1.1952 3.2281 sec/batch
    Epoch 20/20  Iteration 3441/3560 Training loss: 1.1946 3.2178 sec/batch
    Epoch 20/20  Iteration 3442/3560 Training loss: 1.1951 3.2654 sec/batch
    Epoch 20/20  Iteration 3443/3560 Training loss: 1.1949 3.1796 sec/batch
    Epoch 20/20  Iteration 3444/3560 Training loss: 1.1957 3.3175 sec/batch
    Epoch 20/20  Iteration 3445/3560 Training loss: 1.1958 3.2162 sec/batch
    Epoch 20/20  Iteration 3446/3560 Training loss: 1.1959 3.2223 sec/batch
    Epoch 20/20  Iteration 3447/3560 Training loss: 1.1958 3.3087 sec/batch
    Epoch 20/20  Iteration 3448/3560 Training loss: 1.1959 3.4913 sec/batch
    Epoch 20/20  Iteration 3449/3560 Training loss: 1.1962 3.3660 sec/batch
    Epoch 20/20  Iteration 3450/3560 Training loss: 1.1959 3.3217 sec/batch
    Epoch 20/20  Iteration 3451/3560 Training loss: 1.1961 3.2116 sec/batch
    Epoch 20/20  Iteration 3452/3560 Training loss: 1.1959 3.2103 sec/batch
    Epoch 20/20  Iteration 3453/3560 Training loss: 1.1964 3.2775 sec/batch
    Epoch 20/20  Iteration 3454/3560 Training loss: 1.1966 3.1874 sec/batch
    Epoch 20/20  Iteration 3455/3560 Training loss: 1.1970 3.2638 sec/batch
    Epoch 20/20  Iteration 3456/3560 Training loss: 1.1965 3.2114 sec/batch
    Epoch 20/20  Iteration 3457/3560 Training loss: 1.1965 3.2267 sec/batch
    Epoch 20/20  Iteration 3458/3560 Training loss: 1.1966 3.3237 sec/batch
    Epoch 20/20  Iteration 3459/3560 Training loss: 1.1966 3.1764 sec/batch
    Epoch 20/20  Iteration 3460/3560 Training loss: 1.1964 3.2706 sec/batch
    Epoch 20/20  Iteration 3461/3560 Training loss: 1.1958 3.2213 sec/batch
    Epoch 20/20  Iteration 3462/3560 Training loss: 1.1957 3.2223 sec/batch
    Epoch 20/20  Iteration 3463/3560 Training loss: 1.1954 3.2525 sec/batch
    Epoch 20/20  Iteration 3464/3560 Training loss: 1.1953 3.1916 sec/batch
    Epoch 20/20  Iteration 3465/3560 Training loss: 1.1949 3.1803 sec/batch
    Epoch 20/20  Iteration 3466/3560 Training loss: 1.1948 3.2380 sec/batch
    Epoch 20/20  Iteration 3467/3560 Training loss: 1.1946 3.2603 sec/batch
    Epoch 20/20  Iteration 3468/3560 Training loss: 1.1944 3.3150 sec/batch
    Epoch 20/20  Iteration 3469/3560 Training loss: 1.1943 3.2743 sec/batch
    Epoch 20/20  Iteration 3470/3560 Training loss: 1.1941 3.1794 sec/batch
    Epoch 20/20  Iteration 3471/3560 Training loss: 1.1937 3.2523 sec/batch
    Epoch 20/20  Iteration 3472/3560 Training loss: 1.1939 3.2165 sec/batch
    Epoch 20/20  Iteration 3473/3560 Training loss: 1.1935 3.2145 sec/batch
    Epoch 20/20  Iteration 3474/3560 Training loss: 1.1935 3.3166 sec/batch
    Epoch 20/20  Iteration 3475/3560 Training loss: 1.1932 3.1766 sec/batch
    Epoch 20/20  Iteration 3476/3560 Training loss: 1.1929 3.2508 sec/batch
    Epoch 20/20  Iteration 3477/3560 Training loss: 1.1927 3.2176 sec/batch
    Epoch 20/20  Iteration 3478/3560 Training loss: 1.1929 3.2095 sec/batch
    Epoch 20/20  Iteration 3479/3560 Training loss: 1.1928 3.2625 sec/batch
    Epoch 20/20  Iteration 3480/3560 Training loss: 1.1925 3.1841 sec/batch
    Epoch 20/20  Iteration 3481/3560 Training loss: 1.1921 3.2603 sec/batch
    Epoch 20/20  Iteration 3482/3560 Training loss: 1.1920 3.2221 sec/batch
    Epoch 20/20  Iteration 3483/3560 Training loss: 1.1919 3.2068 sec/batch
    Epoch 20/20  Iteration 3484/3560 Training loss: 1.1917 3.3919 sec/batch
    Epoch 20/20  Iteration 3485/3560 Training loss: 1.1916 3.4986 sec/batch
    Epoch 20/20  Iteration 3486/3560 Training loss: 1.1916 3.3447 sec/batch
    Epoch 20/20  Iteration 3487/3560 Training loss: 1.1914 3.2831 sec/batch
    Epoch 20/20  Iteration 3488/3560 Training loss: 1.1912 3.2356 sec/batch
    Epoch 20/20  Iteration 3489/3560 Training loss: 1.1911 3.2129 sec/batch
    Epoch 20/20  Iteration 3490/3560 Training loss: 1.1912 3.3290 sec/batch
    Epoch 20/20  Iteration 3491/3560 Training loss: 1.1909 3.1965 sec/batch
    Epoch 20/20  Iteration 3492/3560 Training loss: 1.1909 3.2663 sec/batch
    Epoch 20/20  Iteration 3493/3560 Training loss: 1.1908 3.2074 sec/batch
    Epoch 20/20  Iteration 3494/3560 Training loss: 1.1907 3.2544 sec/batch
    Epoch 20/20  Iteration 3495/3560 Training loss: 1.1906 3.2906 sec/batch
    Epoch 20/20  Iteration 3496/3560 Training loss: 1.1905 3.1791 sec/batch
    Epoch 20/20  Iteration 3497/3560 Training loss: 1.1904 3.0981 sec/batch
    Epoch 20/20  Iteration 3498/3560 Training loss: 1.1901 3.1790 sec/batch
    Epoch 20/20  Iteration 3499/3560 Training loss: 1.1900 3.2523 sec/batch
    Epoch 20/20  Iteration 3500/3560 Training loss: 1.1900 3.3326 sec/batch
    Epoch 20/20  Iteration 3501/3560 Training loss: 1.1898 3.2638 sec/batch
    Epoch 20/20  Iteration 3502/3560 Training loss: 1.1897 3.2071 sec/batch
    Epoch 20/20  Iteration 3503/3560 Training loss: 1.1896 3.2829 sec/batch
    Epoch 20/20  Iteration 3504/3560 Training loss: 1.1894 3.2171 sec/batch
    Epoch 20/20  Iteration 3505/3560 Training loss: 1.1890 3.2083 sec/batch
    Epoch 20/20  Iteration 3506/3560 Training loss: 1.1889 3.2735 sec/batch
    Epoch 20/20  Iteration 3507/3560 Training loss: 1.1888 3.5122 sec/batch
    Epoch 20/20  Iteration 3508/3560 Training loss: 1.1885 3.3543 sec/batch
    Epoch 20/20  Iteration 3509/3560 Training loss: 1.1885 3.2818 sec/batch
    Epoch 20/20  Iteration 3510/3560 Training loss: 1.1884 3.2559 sec/batch
    Epoch 20/20  Iteration 3511/3560 Training loss: 1.1882 3.1942 sec/batch
    Epoch 20/20  Iteration 3512/3560 Training loss: 1.1878 3.1003 sec/batch
    Epoch 20/20  Iteration 3513/3560 Training loss: 1.1874 3.0324 sec/batch
    Epoch 20/20  Iteration 3514/3560 Training loss: 1.1872 3.0799 sec/batch
    Epoch 20/20  Iteration 3515/3560 Training loss: 1.1873 3.2454 sec/batch
    Epoch 20/20  Iteration 3516/3560 Training loss: 1.1874 3.3141 sec/batch
    Epoch 20/20  Iteration 3517/3560 Training loss: 1.1873 3.4988 sec/batch
    Epoch 20/20  Iteration 3518/3560 Training loss: 1.1874 3.3365 sec/batch
    Epoch 20/20  Iteration 3519/3560 Training loss: 1.1875 3.2654 sec/batch
    Epoch 20/20  Iteration 3520/3560 Training loss: 1.1876 3.2268 sec/batch
    Epoch 20/20  Iteration 3521/3560 Training loss: 1.1876 3.1506 sec/batch
    Epoch 20/20  Iteration 3522/3560 Training loss: 1.1876 3.1006 sec/batch
    Epoch 20/20  Iteration 3523/3560 Training loss: 1.1879 3.1819 sec/batch
    Epoch 20/20  Iteration 3524/3560 Training loss: 1.1880 3.3317 sec/batch
    Epoch 20/20  Iteration 3525/3560 Training loss: 1.1880 3.3284 sec/batch
    Epoch 20/20  Iteration 3526/3560 Training loss: 1.1882 3.3079 sec/batch
    Epoch 20/20  Iteration 3527/3560 Training loss: 1.1881 3.3693 sec/batch
    Epoch 20/20  Iteration 3528/3560 Training loss: 1.1883 3.3131 sec/batch
    Epoch 20/20  Iteration 3529/3560 Training loss: 1.1883 3.2092 sec/batch
    Epoch 20/20  Iteration 3530/3560 Training loss: 1.1885 3.2406 sec/batch
    Epoch 20/20  Iteration 3531/3560 Training loss: 1.1887 3.2364 sec/batch
    Epoch 20/20  Iteration 3532/3560 Training loss: 1.1886 3.2253 sec/batch
    Epoch 20/20  Iteration 3533/3560 Training loss: 1.1884 3.2564 sec/batch
    Epoch 20/20  Iteration 3534/3560 Training loss: 1.1883 3.1813 sec/batch
    Epoch 20/20  Iteration 3535/3560 Training loss: 1.1884 3.2650 sec/batch
    Epoch 20/20  Iteration 3536/3560 Training loss: 1.1883 3.2144 sec/batch
    Epoch 20/20  Iteration 3537/3560 Training loss: 1.1883 3.2073 sec/batch
    Epoch 20/20  Iteration 3538/3560 Training loss: 1.1882 3.2626 sec/batch
    Epoch 20/20  Iteration 3539/3560 Training loss: 1.1882 3.1874 sec/batch
    Epoch 20/20  Iteration 3540/3560 Training loss: 1.1881 3.3095 sec/batch
    Epoch 20/20  Iteration 3541/3560 Training loss: 1.1878 3.4204 sec/batch
    Epoch 20/20  Iteration 3542/3560 Training loss: 1.1880 3.4155 sec/batch
    Epoch 20/20  Iteration 3543/3560 Training loss: 1.1881 3.3240 sec/batch
    Epoch 20/20  Iteration 3544/3560 Training loss: 1.1882 3.2578 sec/batch
    Epoch 20/20  Iteration 3545/3560 Training loss: 1.1881 3.1885 sec/batch
    Epoch 20/20  Iteration 3546/3560 Training loss: 1.1881 3.2863 sec/batch
    Epoch 20/20  Iteration 3547/3560 Training loss: 1.1880 3.2437 sec/batch
    Epoch 20/20  Iteration 3548/3560 Training loss: 1.1880 3.2171 sec/batch
    Epoch 20/20  Iteration 3549/3560 Training loss: 1.1881 3.2884 sec/batch
    Epoch 20/20  Iteration 3550/3560 Training loss: 1.1884 3.5053 sec/batch
    Epoch 20/20  Iteration 3551/3560 Training loss: 1.1884 3.3503 sec/batch
    Epoch 20/20  Iteration 3552/3560 Training loss: 1.1884 3.2899 sec/batch
    Epoch 20/20  Iteration 3553/3560 Training loss: 1.1883 3.2379 sec/batch
    Epoch 20/20  Iteration 3554/3560 Training loss: 1.1883 3.2553 sec/batch
    Epoch 20/20  Iteration 3555/3560 Training loss: 1.1884 3.2623 sec/batch
    Epoch 20/20  Iteration 3556/3560 Training loss: 1.1884 3.1706 sec/batch
    Epoch 20/20  Iteration 3557/3560 Training loss: 1.1885 3.1125 sec/batch
    Epoch 20/20  Iteration 3558/3560 Training loss: 1.1883 3.1610 sec/batch
    Epoch 20/20  Iteration 3559/3560 Training loss: 1.1882 3.2287 sec/batch
    Epoch 20/20  Iteration 3560/3560 Training loss: 1.1884 3.3650 sec/batch
    Validation loss: 1.12251 Saving checkpoint!


#### Saved checkpoints

Read up on saving and loading checkpoints here: https://www.tensorflow.org/programmers_guide/variables


```python
tf.train.get_checkpoint_state('checkpoints')
```




    model_checkpoint_path: "checkpoints/i3560_l512_v1.123.ckpt"
    all_model_checkpoint_paths: "checkpoints/i200_l512_v2.432.ckpt"
    all_model_checkpoint_paths: "checkpoints/i400_l512_v2.012.ckpt"
    all_model_checkpoint_paths: "checkpoints/i600_l512_v1.783.ckpt"
    all_model_checkpoint_paths: "checkpoints/i800_l512_v1.619.ckpt"
    all_model_checkpoint_paths: "checkpoints/i1000_l512_v1.499.ckpt"
    all_model_checkpoint_paths: "checkpoints/i1200_l512_v1.418.ckpt"
    all_model_checkpoint_paths: "checkpoints/i1400_l512_v1.357.ckpt"
    all_model_checkpoint_paths: "checkpoints/i1600_l512_v1.305.ckpt"
    all_model_checkpoint_paths: "checkpoints/i1800_l512_v1.255.ckpt"
    all_model_checkpoint_paths: "checkpoints/i2000_l512_v1.228.ckpt"
    all_model_checkpoint_paths: "checkpoints/i2200_l512_v1.202.ckpt"
    all_model_checkpoint_paths: "checkpoints/i2400_l512_v1.189.ckpt"
    all_model_checkpoint_paths: "checkpoints/i2600_l512_v1.170.ckpt"
    all_model_checkpoint_paths: "checkpoints/i2800_l512_v1.156.ckpt"
    all_model_checkpoint_paths: "checkpoints/i3000_l512_v1.146.ckpt"
    all_model_checkpoint_paths: "checkpoints/i3200_l512_v1.137.ckpt"
    all_model_checkpoint_paths: "checkpoints/i3400_l512_v1.131.ckpt"
    all_model_checkpoint_paths: "checkpoints/i3560_l512_v1.123.ckpt"



## Sampling

Now that the network is trained, we'll can use it to generate new text. The idea is that we pass in a character, then the network will predict the next character. We can use the new one, to predict the next one. And we keep doing this to generate all new text. I also included some functionality to prime the network with some text by passing in a string and building up a state from that.

The network gives us predictions for each character. To reduce noise and make things a little less random, I'm going to only choose a new character from the top N most likely characters.




```python
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
```


```python
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = build_rnn(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)
```

Here, pass in the path to a checkpoint and sample from the network.


```python
checkpoint = "/media/Data_Drive/deep-learning/intro-to-rnns/checkpoints/i3560_l512_v1.123.ckpt"
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Far")
print(samp)
```

    Farening, who, they were at and might attricted them and were at home; and
    then well, and how the princess had talked fix held a little on the
    room, their wife and the courses. But that his clove was at that condition
    too. But, who was talking along this. And while his eyes could not
    be done to him.
    
    Sometimes the sick man the conviccions, and the doctor had settled
    in the sound of this arminary, but he hed softered into some sorts of
    children, and an intention, and her face, the party of the read, and all his
    streight he was an immense arminurily, and he did not know what to stair,
    stopped a lady. At the country was starting, but had an idea, and that
    what it was a mushroom. But he said it. She dropped the poor of his head.
    "It's not thought.... Well, while, and the peasant women."
    
    "Yes, and that's she the life."
    
    "Oh, the servants! They'll get a luttle."
    
    He was saying to him.
    
    "I am voice is straight on it," he said, stopping at him so made and his
    hand to the country. The marshal on the dining room, the possiblion of his
    child with his figure and a latter. Anna went away to her, and he had
    so liked to thought he saw that the peipan of the chasticaties all this
    work it were a long while to anster, this is so the place which she
    had been delighted in the conditions of simplition.
    
    This was this sended the peasants. His short and which how he had an answer
    than the posicial position in the country. The same complete pose the
    peasant came into town or which the caught he had said:
    
    Stiva seemed at the solation of the chird to the diricular carriage as
    he had told him of the same, without the standan, and the convintions
    were not suppering her at the mind as that was no concert of the same
    still.
    
    "I should go and was taken the mote another, a cordear to this man."
    
    "No, you don't us all the time of it."
    
    "What do you too, but I saw that it would seek a legs to think?"
    
    "Well, then at the steps and this count your expensities, but it's a
    perfection," she said to himself

