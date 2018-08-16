"""
Logging and Data Scaling Utilities

Adapted by David Alvarez Charris (david13.ing@gmail.com)

Original code: Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os
import shutil
import glob
import csv


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now, logname_file=''):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
            logname_file: suffix of the name of the log file
        """
        logname = ''.join(str(e) +'/' for e in logname)
        path = os.path.join('log-files', logname, now)
        self.is_loaded_checkpoint = os.path.exists(path) # True if agent is being relodaded from checkpoint (log already exist)

        if not self.is_loaded_checkpoint:  os.makedirs(path) # create directory if it doesn't exist
        # filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        # for filename in filenames:     # for reference
        #     shutil.copy(filename, path)
        path = os.path.join(path, 'log{}.csv'.format(logname_file))
        print("Path for Saved Logs: {}".format(path))

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'a')
        self.writer = None  # DictWriter created with first call to write() method
        self.logname_file = logname_file
        self.log_path = path

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry, self.logname_file)
        if self.write_header:
            print("FIELDS")
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            if not self.is_loaded_checkpoint: self.writer.writeheader() # Only if logger does not exist, write a row with the Headers
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log, logname_file):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** MTL{}: Time Step {}, Mean R = {:.1f} *****'.format(logname_file, log['_TimeStep'], log['_MeanReward']))

        print('{:s}: {:.3g}'.format('PolicyLoss', log['PolicyLoss']))
        print('{:s}: {:.3g}'.format('ValFuncLoss', log['ValFuncLoss']))
        # for key in log_keys:
        #     if key[0] != '_':  # don't display log items with leading '_'
        #         print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
