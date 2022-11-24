import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['figure.figsize'] = 6, 4
rcParams['xtick.bottom'] = True
rcParams['ytick.left'] = True

def visualize_toy_regression(x, y, x_test, y_test, mean_preds, std_preds, save_path=''):
    '''
        args:
            x: train inputs
            y: train labels
            x_test: test inputs
            y_test: test labels
            mean_preds: mean predictions
            std_preds: std predictions
    '''
    plt.fill_between(x_test, mean_preds - 5 * std_preds, mean_preds + 5 * std_preds,
                        color='cornflowerblue', alpha=.5, label='+/- 5 std')
    plt.scatter(x, y,marker='x', c='black', label='target', alpha=0.25)
    plt.plot(x_test, mean_preds, c='red', label='Prediction')
    plt.plot(x_test, y_test, c='grey', label='truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, facecolor='w', dpi=300,bbox_inches='tight')
    plt.clf()

def visualize_toy_classification(xx, yy, X, Y, entropy, save_path):
    # range size for mesh

    # X0 = X[:, 0]
    # X1 = X[:, 1]

    # # Find the range of the 2 dimensions that we will plot
    # X0_min, X0_max = X0.min()-1, X0.max()+1
    # X1_min, X1_max = X1.min()-1, X1.max()+1

    # n_steps = 100 # Number of steps on each axis

    # # Create a meshgrid
    # xx, yy = np.meshgrid(np.arange(X0_min, X0_max, (X0_max-X0_min)/n_steps),
    #                     np.arange(X1_min, X1_max, (X1_max-X1_min)/n_steps))
    
    # Z = preds_proba
    # Z = entropy(Z, axis=1) 
    # Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, entropy, cmap=plt.cm.coolwarm, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.colorbar()
    plt.savefig(save_path, facecolor='w', dpi=300,bbox_inches='tight')
    # plt.show()

