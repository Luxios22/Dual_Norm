import re
import os

def get_last_stats(output_dir, seek=['epo'], epo=-1):
    p = re.compile(r'epo([\d]+).pth')
    files = [(f, int(p.search(f).groups()[0]))
             for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)) and p.search(f) is not None]

    if epo == -1:
        files.sort(key=lambda t: t[1], reverse=True)
        last_epo = files[0][1]
        checkpoints = {'epo': last_epo}
        for file in files:
            if file[1] != last_epo:
                break
            for name in seek:
                if name in file[0] and name!='epo':
                    checkpoints[name] = os.path.join(output_dir, file[0])
        return checkpoints
    else:
        checkpoints = {'epo': epo}
        for file in files:
            if file[1] != epo:
                continue
            for name in seek:
                if name in file[0] and name!='epo':
                    checkpoints[name] = os.path.join(output_dir, file[0])
        return checkpoints

if __name__ == '__main__':
    print(get_last_stats('/home/tangyingtian/Person_ReID_Cross_Domain/checkpoint/Cross_domain/baseline_tuned/'))
