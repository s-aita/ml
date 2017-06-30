#!/usr/bin/env python

from optparse import OptionParser
import math
import random
import numpy as np
# import cupy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

def target_func1(x):
    return 4.0 * (x - 0.5) * (x - 0.5)
def target_func2(x):
    return math.cos(math.pi * x) / 2.0 + 0.5
def target_func(x):
    return target_func2(x)

class MyChain2(Chain):
    def __init__(self, n_hidden):
        super(MyChain2, self).__init__(
            l1=L.Linear(None, n_hidden),
            l2=L.Linear(None, n_hidden),
            l3=L.Linear(None, 2),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        y = F.sigmoid(self.l3(h2))
        return y

class Classifier(Chain):
  def __init__(self, predictor):
    super(Classifier, self).__init__()
    with self.init_scope():
      self.predictor = predictor

  def __call__(self, x, t):
    y = self.predictor(x)
    loss = F.softmax_cross_entropy(y, t)
    accuracy = F.accuracy(y, t)
    return loss

def clamp(minimum, x, maximum):
    return sorted((minimum, x, maximum))[1]

def get_batch(n):
  inputs = np.zeros([n, 2], dtype=np.float32)
  labels = np.zeros([n], dtype=np.int32)
  sigma2 = 0.1
  for i in range(n):
    x_in = random.random()
    y_in = random.random()
    y_border = target_func(x_in)
    if y_border > y_in:
      labels[i] = 0
    else:
      labels[i] = 1
    x_noise = clamp(0.0, random.gauss(x_in, sigma2), 1.0)
    y_noise = clamp(0.0, random.gauss(y_in, sigma2), 1.0)
    inputs[i][0] = x_noise
    inputs[i][1] = y_noise
  return [Variable(inputs), Variable(labels)]

def predict(model, x, y):
    # first dimension is data (only one set)
    v = Variable(np.array([[x, y]], dtype=np.float32))
    labels = model.predictor(v)
    return np.argmax(labels.data)

def test(model):
    f = 20.0
    n_all = 0
    n_correct = 0
    for yi in range(round(f)):
        y = 1.0 - (yi / f)
        results = []
        for xi in range(round(f)):
            x = xi / f
            y_border = target_func(x)
            label = predict(model, x, y)
            if y_border <= y:
                label += 2
            n_all += 1
            if label == 0 or label == 3:
                n_correct += 1
            results.append(str(label))
        print(''.join(results))
    print({'n_all': n_all, 'n_correct': n_correct, 'correct%': 100.0 * n_correct / n_all})

if __name__ == '__main__':
    option_parser = OptionParser()
    option_parser.add_option('-n', '--n-per-hidden', type='int', metavar='N', dest='n_hidden', default=30, help='number of hidden units per layer')
    option_parser.add_option('-e', '--n-episode', type='int', metavar='N', dest='n_episode', default=10000, help='number of episodes')
    option_parser.add_option('-b', '--n-batch', type='int', metavar='N', dest='n_batch', default=30, help='number of samples per batch')
    (options, args) = option_parser.parse_args()

    model = L.Classifier(MyChain2(options.n_hidden))
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # model.to_gpu(0)

    print('------ before training ------')
    test(model)

    for _ in range(options.n_episode):
        inputs, labels = get_batch(options.n_batch)
        optimizer.update(model, inputs, labels)

    print('------ after training ------')
    test(model)
