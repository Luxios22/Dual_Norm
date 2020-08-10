from __future__ import print_function
import warnings
warnings.filterwarnings('ignore','.*conversion.*')
import os
import zipfile
import shutil
import requests
import numpy as np
import argparse

import warnings
from sys import stdout
from os import makedirs
from os.path import dirname
from os.path import exists

################################
# Dataset with Google Drive IDs#
################################

dataset = {
    'VIPeR': '1Kj2cQuVFsgK6vZZI1FV8rvNny4hOU2hi',
    'ETHZ':'1qf2MCJX-hW1NNCVckT5ZVy3xbx-sAqjb',
    'PRID2011':'1EIdmr9LAKLUgZJlWvCaLHcRRLNSgb34R',
    'GRID':'1uy001HaC7cdZ1bHTKoLlV4skCQ08qZWB',
    'CAVIAR4ReID':'1z2BAY9VpvJOZ-tEqJRluztcOVpaTsAuQ',
    '3DPeS':'1JzlH4xcZlnuB7ec8CxN4YlTSu3TvnYsh',
    '3DPeS_RawVideo':'1Qp99Gu6aSp0GkZPxdksgFiL3xDHzMTVu',
    'V-47':'10WIK2HV5UJzsRO_aJGdbT0iTxYq0I1sJ',
    'i-LIDS':'1XXvmL8TuwLxnvVaFyA-BX3O6fb4-1ptK',
    'iLIDS-VID':'1dAAPCJLDqmi7__uKZG5z-Kgl2Z7Wavyd',
    'WARD':'1NzMldJP0976sW70PT8L2wiwcdTwZr80l',
    'SAIVT-SoftBio':'1v6JtHAat4q5CEFkwUY0FrG3xQE8mae5E',
    'SAIVT-SoftBio_Segmentation':'1WSNvmfYkr63r9Eu8Q4WUM-21tVc96GR0',
    'CUHK01':'1YTnHNQEcoXy9iBHkxDsdfzRwqdRHdIVJ',
    'CUHK02':'1nHbmrWOJtls-iWjyRJgl-MtFuwpl_pMF',
    'CUHK03':'16-jnTygO2-PGOO7bSgsUkqwl1zcJl9fm',
    'RAiD':'1m3bs4f316JQlfc97XG-iM1TAVIOUxrPE',
    'CUHK-SYSU':'1E01BB1IbdTxh6ZpclMMio8hkYvXAHiGx',
    'MARS':'16PoINVLW-swqi-tEsOqpedrq-QZuIvfi',
    'CUHK-PEDES':'1-_vPNBBdjwYOVYofavsQX451_l4JXK-q',
    'PRW':'106Qv6DC5ymTv8m5O59Pbjdz--Qyk97bX',
    'Market-1501': '1RJC5YSpRpbA6vHv_zdvGZrK60vvsCMCv',
    'Market-1501_Attribute':'10VbiDHtHeBBY13KbclWv7EQ6M4ZHZIru',
    'PKU-Reid':'10ztvPY3jvIBdlKZGuOzFTWXvUdrJSy1i',
    'DukeMTMC-reID': '1A2eM9CUd8KeIEzpltGxEFeKUXnIoynbn',
    'PRAI-1581': '11shGm0PSNheSRVksO1rBIIkVNR86AzDm',
    'DukeMTMC-reID_Attribute':'1DSBm7awKKZ8bnvWrr9G4didRrnTUo1Y0',
    'Airport':'1IBNJB5iJ72CqQFmFwUS4oLneWdkOFg9m',
    'MSMT17_V2':'1BTgKwxqXC4HM6_xXht82JOzCGMD3gkDk',
    'RPIfield':'1GQMtFXeRAlOlyE5NjIxuv5Ng_wldjTq9',
    'LPW':'14CEgAM410_Ekg0MMLLngOks-WwyhUw-b',
    'market-white':'1sxQmEasWyNazHhA1l1WzI2c_7V9MpZh9',
    'market-green':'18njISYENxphhuySvHi5J576a4UTqA3hf',
    'market-blue':'1ByuRTzau7o8BlAB1h7NIQQwB0zyeFDja',
    'market-black': '1uPo1h_OkuKsXsF4ZW1O3u-gOdm3d6m-4'
}

##########################
# Google Drive Downloader#
##########################
class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, showsize=True):
        """
        Downloads a shared file from google drive into a given folder.

        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        showsize: bool
            optional, if True print the current download size.
        Returns
        -------
        None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()
            stdout.flush()
            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(response, dest_path, showsize, current_download_size)
            print('Done.')


    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print('\r' + GoogleDriveDownloader.sizeof_fmt(current_size[0]), end=' ')
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    # From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Yi', suffix)

###########################
# ReID Dataset Downloader#
###########################

def Person_ReID_Dataset_Downloader(save_dir, dataset_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_exist = os.path.join(save_dir , dataset_name)

    if not os.path.exists(save_dir_exist):
        temp_dir = os.path.join(save_dir , 'temp')

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        destination = os.path.join(temp_dir , dataset_name)

        id = dataset[dataset_name]

        print("Downloading %s" % dataset_name)
        # gdrive_downloader(destination, id)
        GoogleDriveDownloader.download_file_from_google_drive(file_id=id,dest_path=destination,showsize=True)

        zip_ref = zipfile.ZipFile(destination)
        print("Extracting %s" % dataset_name)
        zip_ref.extractall(save_dir)
        zip_ref.close()
        shutil.rmtree(temp_dir)
        print("Done")
    else:
        print("Dataset Check Success: %s exists!" %dataset_name)

#For Unit Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="save_dir", action="store", default="~/Temp/",help="")
    parser.add_argument(dest="dataset_name", action="store", default="PRAI-1581", type=str,help="")
    args = parser.parse_args()
    Person_ReID_Dataset_Downloader(args.save_dir,args.dataset_name)
