import config

import numpy as np
import numpy.linalg as LA

from src.attackers.fast_gradient import FastGradientMethod
import tensorflow as tf

supported_methods = {"fgsm": {"class": FastGradientMethod,
                              "params": {"eps_step": 0.1, "eps_max": 1., "clip_min": 0., "clip_max": 1.}},
                      # "jsma": {"class": SaliencyMapMethod,
                      #          "params": {"theta": 1., "gamma": 0.01, "clip_min": 0., "clip_max": 1.}}
                      }


def get_crafter(method, model, session, params=None):

    try:
        crafter = supported_methods[method]["class"](model, sess=session)
    except:
        raise NotImplementedError("{} crafting method not supported.".format(method))

    if params:
        crafter.set_params(**params)
    else:
        crafter.set_params(**supported_methods[method]["params"])

    return crafter


def empirical_robustness(x, model, sess, method_name, method_params=None):
    """ Computes the Empirical Robustness of a `model` over the sample `x` for a given adversarial crafting method 
    `method_name`, following https://arxiv.org/abs/1511.04599
    
    :param x: 
    :param model: 
    :param method_name: 
    :param sess: 
    :param method_params: 
    :return: 
    """

    crafter = get_crafter(method_name, model, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    perts_norm = LA.norm((adv_x-x).reshape(x.shape[0], -1), ord=crafter.ord, axis=1)

    assert perts_norm.shape == (len(x), ), perts_norm.shape

    return np.mean(perts_norm/LA.norm(x))


def kernel_rbf(x,y,sigma=0.1):
    """Computes the kernel 

    :param x: a tensor object or a numpy array
    :param y: a tensor object or a numpy array

    returns: a tensor object
    """
    norms_x = tf.reduce_sum(x ** 2, 1)[:, None] # axis = [1] for later tf vrsions
    norms_y = tf.reduce_sum(y ** 2, 1)[None, :]
    dists = norms_x - 2 * tf.matmul(x, y, transpose_b=True) + norms_y
    return tf.exp(-(1.0/(2.0*sigma)*dists))


def mmd(x_data,y_data,sess,sigma=0.1):
    """ Computes Maximum Mean Disrepancy between x and y

    :param x_data: numpy array
    :param y_data: numpy array

    returns: a float value corresponding to mmd(x_data,y_data)
    """
    assert x_data.shape[0]==y_daya.shape[0]
    x_data = x_data.reshape(x_data.shape[0],np.prod(x_data.shape[1:]))
    y_data = y_data.reshape(y_data.shape[0],np.prod(y_data.shape[1:]))
    x = tf.placeholder(tf.float32, shape=x_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)
    mmd = tf.reduce_sum(kernel_rbf(x,x)) - \
            2.0*tf.reduce_sum(kernel_rbf(x,y)) + tf.reduce_sum(kernel_rbf(y,y))
    
    return sess.run(mmd, feed_dict = {x:x_data, y:y_data})


def mmd_metric(x, model, sess, method_name, method_params=None):
    """ 
    
    """
    crafter = get_crafter(method_name, model, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    return mmd(x,adv_x,sess)


def nn_dist(x, model, x_train,  sess, method_name, method_params=None):
    """
    Nearest Neighbour distance
    """
    crafter = get_crafter(method_name, model, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    min_dist = tf.reduce_min(kernel_rbf(x,x_train),reduce_indices=1)
    avg_mean_dist = tf.reduce_min(min_dist)
    return sess.run(avg_mean_dist)

def stoch_preds(x,model,sess):
    """
    TODO 
    """
    y = model(x)
    if x.shape[0] <= 100:
        pass
    else:
        #run batch
        pass

def mc_drop(x, model, sess, method_name, method_params=None):

    '''
    TODO
    droptout at test time for the crafted adversarial examples
    '''
   
    crafter = get_crafter(method_name, model, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    pass
    