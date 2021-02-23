import argparse
import os
import urllib.request

class SquadDownloader:
    def __init__(self, save_path):
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(self.save_path + '/squad_v1.1'):
            os.makedirs(self.save_path + '/squad_v1.1')

        if not os.path.exists(self.save_path + '/squad_v2.0'):
            os.makedirs(self.save_path + '/squad_v2.0')

        self.download_urls = {
            'https://rajpurkar.github.io/SQuAD-explorer' '/dataset/train-v1.1.json': 'squad_v1.1/train.json',
            'https://rajpurkar.github.io/SQuAD-explorer' '/dataset/dev-v1.1.json': 'squad_v1.1/dev.json',
            'https://rajpurkar.github.io/SQuAD-explorer' '/dataset/train-v2.0.json': 'squad_v2.0/train.json',
            'https://rajpurkar.github.io/SQuAD-explorer' '/dataset/dev-v2.0.json': 'squad_v2.0/dev.json',
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
    parser = argparse.ArgumentParser(description='Download Squad')
    parser.add_argument(
        '--download_dir',
        type=str,
        required=False,
        help='directory to store data',
        default="./data",
    )
    args = parser.parse_args()
    print("Download Directory is {}".format(args.download_dir))
    squad_dl = SquadDownloader(args.download_dir)
    squad_dl.download()