import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 6, 4
rcParams['xtick.bottom'] = True
rcParams['ytick.left'] = True

def visualize_toy(x, y, x_test, y_test, mean_preds, std_preds, save_path=''):
    '''
        args:
            x: train inputs
            y: train labels
            x_test: test inputs
            y_test: test labels
            mean_preds: mean predictions
            std_preds: std predictions
    '''
    plt.fill_between(x_test, mean_preds - 3 * std_preds, mean_preds + 3 * std_preds,
                        color='cornflowerblue', alpha=.5, label='+/- 3 std')
    plt.scatter(x, y,marker='x', c='black', label='target')
    plt.plot(x_test, mean_preds, c='red', label='Prediction')
    plt.plot(x_test, y_test, c='grey', label='truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, facecolor='w', dpi=300,bbox_inches='tight')
    plt.clf()


