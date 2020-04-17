import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_weight(shape, name):
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)


def dense(inputs, weights, dropout=0.1):
    y = tf.matmul(inputs, weights)
    y = tf.nn.relu(y)
    if dropout > 0:
        y = tf.nn.dropout(y, rate=dropout)
    return y


# Load data
input = np.clip(np.load('data/spectrum_label.npy').astype(np.float32), a_min=1, a_max=None)
target = np.clip(np.load('data/spectrum_target.npy').astype(np.float32), a_min=1, a_max=None)
# Defining train and test datasets
train_fraction = 0.8
input_train, target_train = input[:int(train_fraction * len(input))], target[:int(train_fraction * len(input))]
input_test = tf.constant(input[int(train_fraction * len(input)):], dtype=tf.float32)
target_test = tf.constant(target[int(train_fraction * len(input)):], dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))

n_in = input.shape[1]  # Number of binds in my data
initializer = tf.initializers.glorot_normal()

weights = [get_weight((n_in, 1000), name='w1'),  # weight of the 1st layer
           get_weight((1000, 2500), name='w2'),  # 1000 because is the shape of w1
           get_weight((2500, 500), name='w3'),
           get_weight((500, n_in), name='w4')]


def model(x):  # x = data (number of events for Erec)

    y1 = dense(x, weights[0])
    y2 = dense(y1, weights[1])
    y3 = dense(y2, weights[2])
    y4 = dense(y3, weights[3], dropout=0) + 1e-3

    return y4


def loss(pred, target):  # prediction = output from the model, target = E_MC
    return tf.reduce_mean(tf.square(tf.math.log(pred) - tf.math.log(target)))  # Analogue to Chi2 method


summary_writer = tf.summary.create_file_writer('logs/train')

optimizer = tf.optimizers.Adam()


def train_step(model, inputs, target):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs), target)
    grads = tape.gradient(current_loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    return tf.reduce_mean(current_loss)


num_epochs = 50
batch_size = 64  # Number of samples taken separately for weight updates
dataset = train_dataset.shuffle(1024).batch(batch_size)
for e in range(num_epochs):
    for features in dataset:
        input, target = features
        _loss = train_step(model, input, target)
    with summary_writer.as_default():
        tf.summary.scalar('loss', _loss, step=e)
        tf.summary.scalar('loss test', loss(model(input_test), target_test), step=e)

y_model = model(input_test)
for i in range(10):
    plt.scatter(np.arange(n_in), input_test[i], label='input')
    plt.scatter(np.arange(n_in), target_test[i], label='target')
    plt.scatter(np.arange(n_in), y_model[i], label='output model')
    plt.yscale('log')
    plt.ylim([0.5, 2*np.max(target_test[i])])
    plt.legend()
    plt.savefig('spectrum_tf2_%i.png' % i, bbox_inches='tight')
    plt.close()
