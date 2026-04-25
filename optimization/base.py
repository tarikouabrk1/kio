from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    
    @abstractmethod
    def search(self, X_train, y_train, X_val=None, y_val=None):
        pass

    def _evaluate(self, model, X_val, y_val, metric):
        preds = model.predict(X_val)
        return metric(y_val, preds)