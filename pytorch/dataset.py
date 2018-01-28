import pandas as pd

class Dataset():
    """
    Loads dataset in csv format
    """
    def __init__(self, MFCC_csv_file, ART_csv_file, transform=None):
        """
        Args:
                csv_file (string): Path to the csv file with annotations
                root_dir (string): Directory with all the images
                transform (callable, optional): Optional transform to be applied
        """
        self.MFCC = pd.read_csv(MFCC_csv_file)
        self.ART = pd.read_csv(ART_csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.ART) # `len(ART)` takes less to compute

    def __getitem__(self, idx):

        sample = None

        MFCC = self.MFCC.iloc[idx, :].as_matrix().astype('float')
        ART = self.ART.iloc[idx, :].as_matrix().astype('float')

        sample = {'MFCC': MFCC, 'ART': ART}

        if self.transform:
            sample = self.transform(sample)

        return sample