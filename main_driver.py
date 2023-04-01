import sys
import warnings
from pandas.core.common import PerformanceWarning

warnings.simplefilter(action='ignore', category=PerformanceWarning)

## Main driver function
def main(args):
    if args[0] == '1':
        from src.challenge1_linreg_house_prices import pipeline1
        pipeline1(args[1])
    
    return

## Main driver
if __name__ == '__main__':
    # Call 'python main_driver.py 1 False' to run challenge 1 with Debug Mode set to False, etc.
    main(sys.argv[1:])