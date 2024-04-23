import helper 
import argparse

def str2bool(val):
    if val in ('t', 'true', '1'):
        return True
    elif val in ('f', 'false', '0'):
        return False
    else:
        print('INVALID BOOLEAN ENTRY')
        return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', type=float)
    parser.add_argument('--mu', type=float)
    parser.add_argument('--lm', type=str)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    reg = args.reg 
    mu = args.mu 
    lm = args.lm 
    mode = args.mode 
    lm = str2bool(lm.lower())

    helper.synthetic_experiments(reg=reg, mu=mu,lm=lm, dir_name='synthetic_test', mode=mode)