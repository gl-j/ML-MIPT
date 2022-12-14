import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    n_objects = y.shape[0]
    y = y.argmax(axis=1)[:, None] #For decoding the one hot representation of y as a 1D ROW vector
    ent = 0.0
    
    #Calculating the probabilities 
    uniques, counts = np.unique(y, return_counts=True)
    dict_probas = {}
    dict_probas = {uniques[i]: counts[i] / n_objects for i in range(len(uniques) ) }
    ent = -1 * np.sum([dict_probas[i] * np.log(dict_probas[i] + EPS) for i in dict_probas.keys() ]) 
    
    return ent
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    n_objects = y.shape[0]
    y = y.argmax(axis=1)[:, None] #For decoding the one hot representation of y as a 1D ROW vector
    
    #Calculating the probabilities 
    uniques, counts = np.unique(y, return_counts=True)
    dict_probas = {}
    dict_probas = {uniques[i]: counts[i] / n_objects for i in range(len(uniques) ) } 
    gini = 1 -  np.sum([dict_probas[i]*dict_probas[i] for i in dict_probas.keys() ])
    
    return gini
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    mean = np.mean(y)
    var = np.sum([(yi - mean) ** 2 for yi in y])
    var *= 1 / len(y)
    return var

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    median = np.median(y)
    mad_median = np.sum([np.abs(yi - median)  for yi in y])
    mad_median *= 1 / len(y)  
    return mad_median


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        n_objects = X_subset.shape[0]
        n_features = X_subset.shape[1]
        n_classes = y_subset.shape[1]   #For regression is only one 
        
        X_left, X_right = np.array([]), np.array([])
        y_left, y_right = np.array([]), np.array([])
    
        for i in range(n_objects): #For each ROW 
            if X_subset[i][feature_index] < threshold:
                X_left = np.append(X_left, X_subset[i] )
                y_left = np.append(y_left, y_subset[i] )
            else:
                X_right = np.append(X_right, X_subset[i] )
                y_right = np.append(y_right, y_subset[i] )
        #Reshaping the resulting arrays to match the original input format 
        X_left.resize( len(X_left) // n_features if len(X_left) // n_features > 0 else (len(X_left) // n_features) + 1, n_features )
        X_right.resize( len(X_right) // n_features if len(X_right) // n_features > 0 else (len(X_right) // n_features) + 1, n_features )
        y_left.resize( len(y_left) // n_classes if len(y_left) // n_classes > 0 else (len(y_left) // n_classes) + 1, n_classes )
        y_right.resize( len(y_right) // n_classes if len(y_right) // n_classes > 0 else (len(y_right) // n_classes) + 1, n_classes )
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        n_objects = y_subset.shape[0]
        n_classes = y_subset.shape[1] #For regression is only one 
        
        y_left, y_right = np.array([]), np.array([])
        
        for i in range(n_objects):
            if X_subset[i][feature_index] < threshold:
                y_left = np.append(y_left, y_subset[i] )
            else:
                y_right = np.append(y_right, y_subset[i] )
        #Reshaping the resulting arrays to match the original input format  
        y_left.resize( len(y_left) // n_classes if len(y_left) // n_classes > 0 else (len(y_left) // n_classes) + 1, n_classes )
        y_right.resize( len(y_right) // n_classes if len(y_right) // n_classes > 0 else (len(y_right) // n_classes) + 1, n_classes )
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        n_objects = X_subset.shape[0]
        n_features = X_subset.shape[1]
    
        if self.criterion_name == "entropy":
            #For entropy the task is classification, therefore a one hot encode of the classes is passed for y_subset  
            info_gain = -1000
            feature_index = -1 
            threshold =-1 
            ent_y = entropy(y_subset )
            #max_entropy = np.log(n_classes) #Maximal theoretical value for the entropy

            for feature in range(n_features):
                uniques = np.unique(X_subset[:, feature]) #Getting the unique values from column/feature 
                for unique in uniques:
                    y_left, y_right = self.make_split_only_y(feature, unique, X_subset, y_subset)
                    ent_left = entropy(y_left )
                    ent_right = entropy(y_right )
                    weighted_av_of_ents = (len(y_left) / n_objects) * ent_left + (len(y_right) / n_objects) * ent_right 
                    current_info_gain = ent_y - weighted_av_of_ents
                    if current_info_gain > info_gain: #This means that the entropy is low , the higher the information gain, the lowest the entropy
                        info_gain = current_info_gain
                        feature_index = feature
                        threshold = unique
        
        elif self.criterion_name == "gini":
            #For entropy the task is classification, therefore a one hot encode of the classes is passed for y_subset  
            info_gain = -1000
            feature_index = -1 
            threshold =-1 
            gini_y = gini(y_subset )
            for feature in range(n_features):
                uniques = np.unique(X_subset[:, feature]) #Getting the unique values from column/feature 
                for unique in uniques:
                    y_left, y_right = self.make_split_only_y(feature, unique, X_subset, y_subset)
                    gini_left = gini(y_left )
                    gini_right = gini(y_right )
                    weighted_av_of_ginis = (len(y_left) / n_objects) * gini_left + (len(y_right) / n_objects) * gini_right 
                    current_info_gain = gini_y - weighted_av_of_ginis
                    if current_info_gain > info_gain: #This means that the gini index is low , the higher the information gain, the lowest the gini index
                        info_gain = current_info_gain
                        feature_index = feature
                        threshold = unique

        elif self.criterion_name == "variance": 
            feature_index = -1 
            threshold =-1 
            current_var = 1000
            for feature in range(n_features):
                uniques = np.unique(X_subset[:, feature]) #Getting the unique values from column/feature 
                for unique in uniques:
                    y_left, y_right = self.make_split_only_y(feature, unique, X_subset, y_subset)
                    var_left = variance(y_left)
                    var_right = variance(y_right)
                    weighted_av_of_vars = (len(y_left) / n_objects) * var_left + (len(y_right) / n_objects) * var_right 
                    if weighted_av_of_vars < current_var: #For minimizing the variance of the split 
                        current_var = weighted_av_of_vars
                        feature_index = feature
                        threshold = unique

        else: #self.criterion = "mad_median": 
            feature_index = -1 
            threshold =-1 
            current_median = 1000
            for feature in range(n_features):
                uniques = np.unique(X_subset[:, feature]) #Getting the unique values from column/feature 
                for unique in uniques:
                    y_left, y_right = self.make_split_only_y(feature, unique, X_subset, y_subset)
                    mad_median_left = mad_median(y_left)
                    mad_median_right = mad_median(y_right)
                    weighted_av_of_mad_medians = (len(y_left) / n_objects) * mad_median_left + (len(y_right) / n_objects) * mad_median_right 
                    if weighted_av_of_mad_medians < current_median: #For minimizing the mad_median of the split 
                        current_median = weighted_av_of_mad_medians
                        feature_index = feature
                        threshold = unique
        
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        
        #For entropy criterion y is given as one hot enocode 
        n_objects = X_subset.shape[0]
        
        if self.criterion_name == "entropy":
            
            if n_objects >= self.min_samples_split and self.depth < self.max_depth: 
                feature_index, threshold = self.choose_best_split(X_subset, y_subset)
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
                
                ent_y = entropy(y_subset) 
                ent_y_left = entropy(y_left)
                ent_y_right = entropy(y_right)
                info_gain = ent_y - ( (len(y_left) / len(y_subset)) * ent_y_left + ( (len(y_right) / len(y_subset)) * ent_y_right ) )
                
                if info_gain > 0.0:
                    #First recursively building left_subtree and left subbranches until the stopping criterion is reached and then it starts building the right subbranches and finally the right subtree 
                    self.depth += 1
                    left = self.make_tree(X_left, y_left)
                    #After reaching a leaf/ creating a leaf recursively building the right subbranches and then the right_subtrees
                    right = self.make_tree(X_right, y_right)
                    
                    #Creating decision node
                    current_node = Node(feature_index, threshold)
                    current_node.left_child = left 
                    current_node.right_child = right 
                    self.depth -= 1 
                    
                    return current_node
            else: #Leaf node , if lean(y_subset) == 1 returns y_subset, else returns the class with the highest ocurrency rate, t.e, the mode 
                new_node = Node(None, None)
                uniques, counts = np.unique(one_hot_decode(y_subset), return_counts=True)
                new_node.value = uniques[counts.argmax()]
                
                #Calculating probas
                probas = np.zeros(self.n_classes)
                if len(y_subset) == 1:
                    probas[int(new_node.value)] = 1
                    new_node.proba = probas 
                else:
                    for (unique, count) in zip(uniques, counts):
                        probas[unique] = count / len(y_subset) 
                    new_node.proba = probas
         
                return new_node
            
        elif self.criterion_name == "gini":
            
            if n_objects >= self.min_samples_split and self.depth < self.max_depth: 
                feature_index, threshold = self.choose_best_split(X_subset, y_subset)
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
                
                gini_y_left = gini(y_left)
                gini_y_right = gini(y_right)           
                weighted_gini = ( (len(y_left) / len(y_subset)) * gini_y_left + ( (len(y_right) / len(y_subset)) * gini_y_right ) )
                if weighted_gini >= 0.0 and weighted_gini <= 1.0:
                    #First recursively building left_subtree and left subbranches until the stopping criterion is reached and then it starts building the right subbranches and finally the right subtree 
                    self.depth += 1 
                    left = self.make_tree(X_left, y_left)
                    #After reaching a leaf/ creating a leaf recursively building the right subbranches and then the right_subtrees
                    right = self.make_tree(X_right, y_right)
                    
                    #Creating decision node 
                    current_node = Node(feature_index, threshold)
                    current_node.left_child = left 
                    current_node.right_child = right 
                    self.depth -= 1 
                    
                    return current_node
            else: #Leaf node, if lean(y_subset) == 1 returns y_subset, else returns the class with the highest ocurrency rate, t.e, the mode 
                new_node = Node(None, None)
                uniques, counts = np.unique(one_hot_decode(y_subset), return_counts=True)
                new_node.value = uniques[counts.argmax()]
                
                #Calculating probas
                probas = np.zeros(self.n_classes)
                if len(y_subset) == 1:
                    probas[int(new_node.value)] = 1
                    new_node.proba = probas
                else: 
                    for (unique, count) in zip(uniques, counts):
                        probas[unique] = count / len(y_subset) 
                    new_node.proba = probas
         
                return new_node

        elif self.criterion_name == "variance":
            
            if n_objects >= self.min_samples_split and self.depth < self.max_depth: 
                feature_index, threshold = self.choose_best_split(X_subset, y_subset)
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
                
                var_y = variance(y_subset) 
                var_left = variance(y_left)
                var_right = variance(y_right)
                info_gain = var_y - ( (len(y_left) / len(y_subset)) * var_left + ( (len(y_right) / len(y_subset)) * var_right ) )

                if info_gain >= 0.0: 
                    #First recursively building left_subtree and left subbranches until the stopping criterion is reached and then it starts building the right subbranches and finally the right subtree 
                    self.depth += 1 
                    left = self.make_tree(X_left, y_left)
                    #After reaching a leaf/ creating a leaf recursively building the right subbranches and then the right_subtrees
                    right = self.make_tree(X_right, y_right)
                    
                    #Creating decision node
                    current_node = Node(feature_index, threshold)
                    current_node.left_child = left 
                    current_node.right_child = right 
                    self.depth -= 1 
                
                    return current_node
            else: #Leaf node, if lean(y_subset) == 1 returns y_subset, else returns the mean of y_subset
                new_node = Node(None, None)
                if len(y_subset) == 1: 
                    new_node.value = y_subset
                else: 
                    new_node.value = np.mean(y_subset)
         
                return new_node

        else: #self.criterion_name == "mad_median":
            
            if n_objects >= self.min_samples_split and self.depth < self.max_depth: 
                feature_index, threshold = self.choose_best_split(X_subset, y_subset)
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
                
                mad_median_y = mad_median(y_subset) 
                mad_median_left = mad_median(y_left)
                mad_median_right = mad_median(y_right)
                info_gain = mad_median_y - ( (len(y_left) / len(y_subset)) * mad_median_left + ( (len(y_right) / len(y_subset)) * mad_median_right ) )
                
                if info_gain > 0.0: 
                    #First recursively building left_subtree and left subbranches until the stopping criterion is reached and then it starts building the right subbranches and finally the right subtree 
                    self.depth += 1 
                    left = self.make_tree(X_left, y_left)
                    #After reaching a leaf/ creating a leaf recursively building the right subbranches and then the right_subtrees
                    right = self.make_tree(X_right, y_right)
                    
                    #Creating decision node
                    current_node = Node(feature_index, threshold)
                    current_node.left_child = left 
                    current_node.right_child = right 
                    self.depth -= 1 

                    return current_node
            else: #Leaf , if lean(y_subset) == 1 returns y_subset, else returns the median of y_subset 
                new_node = Node(None, None)
                if len(y_subset) == 1: 
                    new_node.value = y_subset
                else: 
                    new_node.value = np.median(y_subset) 
                    
                return new_node
        
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    '''Auxiliary method added for recursively compute the prediction of a given instance,
    this method is then used in the method self.predict()'''
    def predict_instance(self, x, sub_tree):
        
        if sub_tree.left_child == None and sub_tree.right_child == None: #sub_tree is A leaf node
            return sub_tree.value 
        
        #If not is a leaf, then if the next threshold condition is satisfied recursively moves to the left,
        #else recursively moves to the right, eventually will reach a leaf node  
        
        #Recursively Moves to the left
        if x[sub_tree.feature_index] < sub_tree.value: #sub_tree.value is the threshold WHEN the node sub_tree is NOT a leaf node
            if sub_tree.left_child: #Is not None    
                return self.predict_instance(x, sub_tree.left_child)
            else:
                return self.predict_instance(x, sub_tree.right_child)
        
        #Recursively Moves to the right
        if x[sub_tree.feature_index] >= sub_tree.value: #sub_tree.value is the threshold WHEN the node sub_tree is NOT a leaf node
            if sub_tree.right_child: #Is not None    
                return self.predict_instance(x, sub_tree.right_child)
            else:
                return self.predict_instance(x, sub_tree.left_child)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        y_predicted = np.array( [self.predict_instance(x, self.root) for x in X ] )
        
        return y_predicted
    
    
    '''Auxiliary method added for recursively compute the prediction of a given instance,
    this method is then used in the method self.predict()'''
    def predict_instance_proba(self, x, sub_tree):
        
        if sub_tree.left_child == None and sub_tree.right_child == None: #(node)sub_tree is A leaf node
            return sub_tree.proba 
        
        #If not is a leaf, then if the next threshold condition is satisfied recursively moves to the left,
        #else recursively moves to the right, eventually will reach a leaf node  
        
        #Recursively Moves to the left
        if x[sub_tree.feature_index] < sub_tree.value: #sub_tree.value is the threshold WHEN the node sub_tree is NOT a leaf node
            if sub_tree.left_child: #Is not None    
                return self.predict_instance_proba(x, sub_tree.left_child)
            else:
                return self.predict_instance_proba(x, sub_tree.right_child)
        
        #Recursively Moves to the right
        if x[sub_tree.feature_index] >= sub_tree.value: #sub_tree.value is the threshold WHEN the node sub_tree is NOT a leaf node
            if sub_tree.right_child: #Is not None    
                return self.predict_instance_proba(x, sub_tree.right_child)
            else:
                return self.predict_instance_proba(x, sub_tree.left_child)
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted_probs = np.array( [self.predict_instance_proba(x, self.root) for x in X ] )
        
        return y_predicted_probs
