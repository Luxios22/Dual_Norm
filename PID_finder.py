import fire
import os
import pandas as pd
from xml.dom import minidom


def finder():
    grabfood100 = pd.read_csv('pid_finder_files/grabfood100.csv')
    xmldoc = minidom.parse('pid_finder_files/all_pedsettings.xml')
    PIDS = []

    hashes = grabfood100['Hash'].values.tolist()
    xml_pedlist = xmldoc.getElementsByTagName('PedSetting')
    for ped in xml_pedlist:
        if ped.attributes['model'].value in hashes:
            PIDS.append(ped.attributes['id'].value)
    # print(PIDS)
    # print(f'{len(PIDS)=}')
    return PIDS

if __name__=='__main__':
    fire.Fire(finder)