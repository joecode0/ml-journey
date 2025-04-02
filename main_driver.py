import sys
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter(action='ignore', category=PerformanceWarning)

## Main driver function
def main(args):
    if args[0] == '1':
        from linreg_house_prices import house_price_pipeline
        house_price_pipeline(args[1])
    
    return

## Main driver
if __name__ == '__main__':
    # Call 'python main_driver.py 1 False' to run challenge 1 with Debug Mode set to False, etc.
    main(sys.argv[1:])