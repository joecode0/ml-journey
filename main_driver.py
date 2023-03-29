## Universal package imports
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

## Custom package imports
from src.challenge1_linreg_house_prices import pipeline1

## Main driver function
def main(args):
    if args[0] == '1':
        pipeline1(args[1])
    
    return

## Main driver
if __name__ == '__main__':
    # Call 'python main_driver.py 1 False' to run challenge 1 with Debug Mode set to False, etc.
    main(sys.argv[1:])