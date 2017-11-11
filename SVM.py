import numpy as np               # use this scientific library for creating & procesing arrays/matrices
import matplotlib.pyplot as plt  # Backend library for plotting
import matplotlib.colors
from matplotlib import style
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd

csv1_name = sys.argv[1]
#csv2_name = sys.argv[2]

#Read the training data
df = pd.read_csv(csv1_name, header=None)
trainingData = (df.iloc[:,1:]).values

labels = (df.iloc[:,0]).values
labels[labels == 0] = -1

#Read the test data
#df = pd.read_csv(csv2_name, header=None)
#testData = (df.iloc[:,:]).values

class SVM(object):
    """
    Support Vector Machine Classifier/Regression
	
    Params: 
    w : float
	weights
    bias : float
	bias
    eta : float
	Learning Rate (0<=eta<1.0)
    epochs : int
	Number of iterations over the training set
	bias : float
	    bias value to initialize w0
	
	Attributes:
	w_ : array
	    Updated weights
	misclassifications_ : list
	    Number of misclassifications in every epoch
    """
	
    def __init__(self, kernel=None, C=None, loss="hinge"):
        self._kernel = kernel
        self._margin = 0
        print ("\n *******************Support Vector Machine Initialization*******************")
        print("Kernel selected ->", kernel)
        if self._C is not None:
            self.C = float(self.C)
            print("\nC ->", C)
        
    #Input the data to this method to train the SVM
    def fit(self, X, y):
        n_samples, n_features = X.shape
        print("Number of examples in a sample = %d, Number of features = %d ", n_samples, n_features)
        self._w = np.zeros(n_features)
        
        # Initialize the Gram matrix for taking the output from QP solution
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        # Here we have to solve the convext optimization problem
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx <= h
        #  Ax = b

        
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        #G & h are required for soft-margin classifier

        if self._C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_std = np.diag(np.ones(n_samples) * -1)
            h_std = np.identity(n_samples)

            G_slack = np.zeros(n_samples)
            h_slack = np.ones(n_samples) * self.C

            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h = cvxopt.matrix(np.hstack((h_std, h_slack)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Now figure out the Support Vectors i.e yi(xi.w + b) = 1
        # Check whether langrange multiplier has non-zero value
        sv = alpha > 0
        self._alpha = alpha[sv]
        self._Support_Vectors = X[sv]
        self._Support_Vectors_Labels = y[sv]

        print ("Total number of examples = %d", n_samples)
        print ("Total number of Support Vectors found = %d", len(self.a))


        #Now we need to find the margin
        ind = np.arange(len(alpha))[sv]
        for n in range(len(self._alpha)):
            self._margin += self._Support_Vectors_Labels[n]
            self._margin -= np.sum(self._alpha * self._Support_Vectors_Labels * K[ind[n], sv])
        self._margin /= len(self._alpha)

        #Get the weight vectors to form the discriminant function
        if (self._kernel == linear_kernel):
            for i in range(len(self._alpha)):
                self._w += self._alpha[i] * self._Support_Vectors_Labels[i] * self._Support_Vectors[i]
        else:
            self._w = None

            
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    def gaussian_kernel(x, y, sigma=5.0):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    def plot_linear_margin(self):
        plt.figure(1)
        plt.subplot(221)
        marker = 'o'

        #we need to make three lines in total

        # w.x + b = 0,  
        a0 = -4;
        a1 = (-self._w[0] * a0 - self._margin ) / self._w[1]
        b0 = 4;
        b1 = (-self._w[0] * b0 - self._margin ) / self._w[1]
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1

        # w.x + b = -1
        #labels
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("SVM with soft margin")
        plt.axis("tight")
        plt.show()
        
#Instantiate the class instance
svm = SVM(kernel=linear_kernel)
svm.fit(trainingData, labels)
svm.plot_learning()
