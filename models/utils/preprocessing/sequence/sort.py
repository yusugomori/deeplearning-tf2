'''
Original Source from:

https://github.com/yusugomori/tftf/blob/master/tftf/preprocessing/sequence/sort.py
'''


def sort(data, target,
         order='ascend'):
    if order == 'ascend' or order == 'ascending':
        a = True
    elif order == 'descend' or order == 'descending':
        a = False
    else:
        raise ValueError('`order` must be of \'ascend\' or \'descend\'.')

    lens = [len(i) for i in data]
    indices = sorted(range(len(lens)),
                     key=lambda x: (2 * a - 1) * lens[x])
    data = [data[i] for i in indices]
    target = [target[i] for i in indices]

    return (data, target)
