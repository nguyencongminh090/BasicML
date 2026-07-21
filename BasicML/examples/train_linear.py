from numpy import gradient
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(suppress=True, precision=4)


from basicml.nn.linear      import LinearRegression
from basicml.nn.loss        import MSELoss
# from basicml.optim.sgd import SGD
from basicml.optim.momentum import Momentum

def main():
    # Load data
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data.csv')
    data      = pd.read_csv(data_path)
    
    X = data[['X']].values
    Y = data[['Y']].values

    # Hyperparam
    epochs = 200
    
    # ---
    model = LinearRegression(features=1)
    loss  = MSELoss()
    optim = Momentum(model.parameters(), 0.05)
    for epoch in range(epochs):
        y_pred = model(X)
        l      = loss(y_pred, Y)
        grad   = loss.backward()
        model.backward(grad)
        optim.step()
        optim.zero_grad()
        print(f'EPOCH: {epoch} | COST: {l} | grad: \n{grad}')
    
    print(f'f(x) = {model.parameters()[0].data.reshape(1,)[0]:.4f}x'\
              f' + {model.parameters()[1].data.reshape(1,)[0]:.4f}')
        
    pass

if __name__ == '__main__':
    main()
