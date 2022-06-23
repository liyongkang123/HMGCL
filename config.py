import argparse
def parse():
    p = argparse.ArgumentParser("HMGNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cuda', type=str, default='0', help='gpu')
    p.add_argument('--city', type=str, default='NYC', help='dataset name (e.g.NYC,TKY,IST,SP,KL,JK)')
    p.add_argument('--epochs', type=int, default=5000, help='number of epochs to train')
    p.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    p.add_argument('--multihead', type=int, default=5, help='number of multiheads')
    p.add_argument('--lambda_1', type=int, default=1, help='number of lambda_1')
    p.add_argument('--lambda_2', type=int, default=2, help='number of lambda_2')
    p.add_argument('--lambda_3', type=int, default=3, help='number of lambda_3')

    args = p.parse_args()
    return args