# Decision Tree Learning using Information Gain, and Chi-Squared Pruning.
# vsingh@uic.edu 07dec17
from pprint import pprint
import numpy as np

# from scipy import stats       # <-- Used for chi-squared pruning

# Building attributes (from example in AI book by Russel&Norvig, fig:18.3)
x0 = [1,1,0,1,1,0,0,0,0,1,0,1]  # Alt
x1 = [0,0,1,0,0,1,1,0,1,1,0,1]  # Bar
x2 = [0,0,0,1,1,0,0,0,1,1,0,1]  # Fri
x3 = [1,1,0,1,0,1,0,1,0,1,0,1]  # Hun
x4 = [1,2,1,2,2,1,0,1,2,2,0,2]  # Pat: 0=None, 1=Some, 2=Full
x5 = [2,0,0,0,2,1,0,1,0,2,0,0]  # Price: 0=$, 1=$$, 2=$$$
x6 = [0,0,0,0,0,1,1,1,1,0,0,0]  # Rain
x7 = [1,0,0,0,1,1,0,1,0,1,0,0]  # Res
x8 = [0,1,2,1,0,3,2,1,2,3,1,2]  # Type: 0=French, 1=Thai, 2=Burger, 3=Italian
x9 = [0,2,0,1,3,0,0,0,3,1,0,2]  # Est: 0=0-10, 1=10-30, 2=30-60, 3= >60

# Target
y = np.array([1,0,1,1,0,1,0,1,0,0,0,1])     # WillWait

# Returns dict with a mapping (k-v) of unique values to their respective..
# ..indices in attribute data.
def get_occurences_per_unique_val(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}

def get_count_dict(a):
    return {c: len((a == c).nonzero()[0]) for c in np.unique(a)}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def information_gain(y, x):
    res = entropy(y)

    # Split x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def DTL(x, y):
    # If there could be no get_occurences_per_unique_val, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    best_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the
    # original set
    if np.all(gain < 1e-6):
        return y

    # We get_occurences_per_unique_val using the selected best attribute
    best_attr_data = get_occurences_per_unique_val(x[:, best_attr])

    res = {}
    for k, v in best_attr_data.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (best_attr, k)] = DTL(x_subset, y_subset)

    # Perform chi-squared pruning
    # PS: Uncomment after installing Scipy module
    # ref: http://scipy.github.io/devdocs/building/index.html
    '''should_prune(res)'''

    return res

# Test driver
X = np.array([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]).T
pprint(DTL(X, y))

# Test o/p
# Tree representation using dictionaries, as follows:
'''
{'x_4 = 0': array([0, 0]),      <-- No Patrons : Do not wait
 'x_4 = 1': array([1, 1, 1, 1]),    <-- Some Patrons: Always wait
 'x_4 = 2': {'x_8 = 0': array([0]), <-- Full: Check Type    <-- French: Do not wait
             'x_8 = 1': {'x_2 = 0': array([0]), 'x_2 = 1': array([1])},
             'x_8 = 2': {'x_0 = 0': array([0]), 'x_0 = 1': array([1])},
             'x_8 = 3': array([0])}}
'''




p_thresh = 0.01

# Performs chi-squared pruning (bottom-up)
# PS: Not tested
def should_prune(tree):
    # Return if leaf
    if type(tree) is not dict:
        return False

    can_prune = True
    attr = ''

    # Consider for pruning only if all children are leaves
    for current_node, sub_tree in tree.items():
        if attr == '':
            attr = int(current_node.split('_')[1][0])

        if type(sub_tree) is dict:
            # current_node cannot be pruned
            can_prune = False
            if should_prune(sub_tree):
                tree[current_node] = "Pruned"

    if not can_prune:
        return False

    if (get_chi_squared_p_val(attr, y) < p_thresh):
        return True

    return False

def get_chi_squared_p_val(x, y):
    labels_with_counts = get_count_dict(y)
    attributes_with_counts = get_count_dict(x)
    total_count = len(y)
    
    chi = 0.0
    for attr_val, attr_count in attributes_with_counts:
        attr_val_freq = attr_count/total_count

        # Get label frequencies per class (Actual) for this attribute value (ex: It's friday)
        label_counts_for_attr_val = get_count_dict(y[x == attr_val])

        # For each label, get the expected value and calculate chi-squared for respective attribute val
        for label, actual_label_count in label_counts_for_attr_val:
            expected_label_count = attr_val_freq * labels_with_counts[label]
            chi += (actual_label_count - expected_label_count)**2 / expected_label_count

    # Calculate p value from chi CDF (replace with commented code after installing scipy)
    p_value = 1     
    # p_value = 1 - stats.chi2.cdf(chi, df=((len(attributes_with_counts)-1)*(len(labels_with_counts)-1)))
    return p_value









