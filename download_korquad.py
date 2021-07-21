import argparse
import os
import urllib.request

class KorquadDownloader:
    def __init__(self, save_path):
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(self.save_path + '/korquad_v1.0'):
            os.makedirs(self.save_path + '/korquad_v1.0')

        # if not os.path.exists(self.save_path + '/korquad_v2.0'):
        #     os.makedirs(self.save_path + '/korquad_v2.0')

        self.download_urls = {
            'https://korquad.github.io' '/dataset/KorQuAD_v1.0_train.json': 'korquad_v1.0/train.json',
            'https://korquad.github.io' '/dataset/KorQuAD_v1.0_dev.json': 'korquad_v1.0/dev.json',
            # 'https://github.com/korquad/korquad.github.io/tree/master' '/dataset/KorQuAD_2.1/train': 'korquad_v2.0/',
            # 'https://github.com/korquad/korquad.github.io/tree/master' '/dataset/KorQuAD_2.1/dev': 'korquad_v2.0/',
        }

    def download(self):
        for item in self.download_urls:
            url = item
            file = self.download_urls[item]

            print('Downloading: %s', url)
            if os.path.isfile(self.save_path + '/' + file):
                print('** Download file already exists, skipping download')
            else:
                response = urllib.request.urlopen(url)
                with open(self.save_path + '/' + file, "wb") as handle:
                    handle.write(response.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download KorQuad')
    parser.add_argument(
        '--download_dir',
        type=str,
        required=False,
        help='directory to store data',
        default="./data",
    )
    args = parser.parse_args()
    print("Download Directory is {}".format(args.download_dir))
    korquad_dl = KorquadDownloader(args.download_dir)
    korquad_dl.download()