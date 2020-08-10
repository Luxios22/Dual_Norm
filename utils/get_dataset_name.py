# Names are unique identifiers in the paths.
names = ['CUHK03',
         'Market1501',
         'DukeMTMC',
         'MSMT17',
         'NTUIndoor',
         'VIPeR',
         'GRID',
         'PRID',
         'QiLIDS',
         'CUHK02',
         'CUHK-SYSU']


def get_dataset_name(pth):
    ls = pth.split(r'/')
    for name in names:
        if name in ls:
            return name
    raise NameError('Unrecognized path: {}. Please update names in get_dataset_name.py.'.format(pth))
