import numpy as np


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    
    
    def _pdf(self, class_idx, x):
        """" Calculate the PDF of x for a given class index """
        
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var) 
        return numerator / denominator
    
    
    def fit(self, X, y):
        """" Fit the Naive Bayes model to the training data """
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Stats for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            # 1. Filter X to only include samples where y == c
            X_c = X[y == c]
            
            # 2. Calculate the stats for this specific class
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / np.float64(n_samples)
        
    
    def predict(self, X):
        """" Predict the class labels for the input data """
        
        predictions = np.array([self._predict(x) for x in X])
        return predictions

    
    def _predict(self, x):
        """" Predict the class label for a single sample x """
        
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)+ 1e-12))
            posterior = prior + class_conditional
            posteriors.append(posterior) 
        
        return self.classes[np.argmax(posteriors)] # Selects class with highest log(P(class|x))