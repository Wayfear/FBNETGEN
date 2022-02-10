import argparse
import re

def main(args):
    table = []
    with open(args.path, 'r') as f:
        lines = f.readlines()

        for l in lines:
            value = re.findall(r'.*Epoch\[(\d+)/500\].*Train Loss: (\d+\.\d+).*Test Loss: (\d+\.\d+)', l)
            table.append(value[0])
    
    s = f'|Epoch|'
    for i in range(0, 500, 50):
        s += f'{i}|'
    print(s)
    
    for j, name in enumerate(['Train Loss', "Test Loss"]):
        s = f'|{name}|'
        for i in range(0, 500, 50):
            s += f'{table[i][j+1]}|'
        print(s)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='ABCD_fbnetgen_cnn_group_loss_false_sparsity_loss_true_window_8_emb_16', type=str,
                        help='Log file path.')
    args = parser.parse_args()
    main(args)