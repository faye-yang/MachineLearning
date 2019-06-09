'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        digit = data.get_digits_by_label(train_data, train_labels, i)
        mean_val = np.mean(digit, 0)
        means[i,:] = mean_val
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for i in range(10):
        digit = data.get_digits_by_label(train_data, train_labels, i)
        diff = digit-means[i]
        cov = np.matmul(diff.T, diff)/(digit.shape[0]-1)
        cov += 0.01 * np.identity(cov.shape[0])
        covariances[i,:,:] = cov
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = digits.shape[0]
    d = digits.shape[1]
    result = np.zeros((n, 10))
    const = -0.5*d*np.log(2*np.pi)
    for i in range(10):
        det = np.linalg.det(covariances[i])
        inv = np.linalg.inv(covariances[i])
        first = const-0.5*np.log(det)
        for j in range(n):
            temp = digits[j]-means[i]
            second = -0.5*np.matmul(np.matmul(temp.T, inv), temp)
            result[j][i] = first+second
    return result


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    log_gl = generative_likelihood(digits, means, covariances)
    log_py = np.log(1/10)
    log_prior = log_gl+log_py
    px_vector = np.sum(np.exp(log_prior), axis=1)
    px = np.tile(px_vector.T, (log_gl.shape[1], 1)).T
    log_px = np.log(px)
    con_likelihood = log_prior-log_px
    return con_likelihood


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    correct = []
    for i in range(digits.shape[0]):
        correct.append(cond_likelihood[i][int(labels[i])])
    result = np.mean(correct)
    return result

def classify_data(digits, means, covariances, labels):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    most_pr_digits = np.argmax(cond_likelihood, axis=1)
    num_total = len(labels)
    result = (most_pr_digits-labels).tolist()
    count = result.count(0)
    acc = count/num_total
    return acc

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('hw5digits')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Plot the leading eigenvectors of each class
    sigmas = []
    for i in range(covariances.shape[0]):
        eigen_values, eigen_vectors = np.linalg.eig(covariances[i])
        # obtain the index for the largest eigenvalue
        index = np.argmax(eigen_values)
        leading_eigen_vector = eigen_vectors[index]
        sigmas.append(leading_eigen_vector.reshape(8, 8))
    sigmas = np.concatenate(sigmas, 1)
    plt.imshow(sigmas, cmap='gray')
    plt.savefig("plot_cov.png")
    plt.close()

    train_ave_cl = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_ave_cl = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    # Evaluation
    train_classify_data = classify_data(train_data, means, covariances, train_labels)
    test_classify_data = classify_data(test_data, means, covariances, test_labels)
    print("train set -- average conditional log-likelihood is: {}".format(train_ave_cl))
    print("test set -- average conditional log-likelihood is: {}".format(test_ave_cl))
    print("train set -- accuracy is: {}".format(train_classify_data))
    print("test set -- accuracy is: {}".format(test_classify_data))

if __name__ == '__main__':
    main()
