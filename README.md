# Gradient Descent Optimization Techniques
This repo is basically a python implementation of the paper: https://arxiv.org/pdf/1609.04747.pdf

## Batch formation and data loading
CSV files are simply read using python pandas and then converted to batches using the following iterator class:
Following is a custom iterator written from scratch to break the data into batches of a certain size:
```python
class batch():
    def __init__(self, data, bs):
        self.data = data
        self.n = len(data) // bs
        self.bs = bs
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch < self.n:
            start = self.current_batch * self.bs
            end = start  + self.bs
            self.current_batch += 1
            if isinstance(self.data, pd.DataFrame):
                return self.data.iloc[start : end]
            else:
                return self.data[start : end]
        else:
            raise StopIteration

```
## Batch gradient descent

## Stochastic gradient descent

## Momentum

## NAG (Nesterov Accelerated Gradient)

## adagrad

## adadelta

## RMSprop

## Adam
