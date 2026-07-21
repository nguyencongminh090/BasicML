import sys
import os
import numpy as np

# Add the parent directory to the path so we can import basicml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicml.nn.linear import LinearRegression

def main():
    print("Setting up training loop...")
    # TODO: Initialize model, criterion, optimizer and loop over epochs
    pass

if __name__ == '__main__':
    main()
