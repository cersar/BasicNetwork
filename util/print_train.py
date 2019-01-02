import numpy as np
import sys


def get_digits(num):
    return int(np.floor(np.log10(num))) + 1


def print_bar(step,steps_num):
    max_len = 30
    bar_len = min(max_len,steps_num)
    percent = step/steps_num
    bar = '{1:{0:d}d}/{2}['.format(get_digits(steps_num), step, steps_num)
    bar += '=' * int(bar_len*percent)
    if step<steps_num:
        bar += '>'
    bar += '.' * (bar_len-int(bar_len*percent)-1)
    bar += ']'
    sys.stdout.write(bar)


def print_metrics(metrics,current,val_num):
    metrics_info = ''
    for m,val in metrics.items():
        if val is not None:
            if m == 'val_loss' or m == 'val_acc':
                    metrics_info += ' - {}: {:.4f}'.format(m, val / val_num)
            else:
                metrics_info += ' - {}: {:.4f}'.format(m, val / current)
    sys.stdout.write(metrics_info)
    sys.stdout.write('\n')


def print_epoch_info(epoch,epoch_num):
    print('epoch: {1:{0:d}d}/{2}'.format(get_digits(epoch_num),epoch,epoch_num))