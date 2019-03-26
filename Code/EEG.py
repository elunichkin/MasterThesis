import csv
import time
import numpy as np
import scipy.stats


class EEG:
    def __init__(self, file):
        self.timer = time.time()
        self.data = self.__read_file(file)
        self.chunk_sizes = [i*1000 for i in range(1, 11)]
        self.cdf, self.rho = {}, {}

    def __display_time(text):
        def decorator(f):
            def wrapper(self, *args, **kwargs):
                print(text, end=' ')
                result = f(self, *args, **kwargs)
                print('finished in {0} s'.format(round(time.time() - self.timer, 2)))
                return result
            return wrapper
        return decorator

    @__display_time('File reading')
    def __read_file(self, file):
        with open(file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='	')
            raw_data = [row for row in csvreader]
            headers = raw_data.pop(0)

        return {headers[i]: np.array([float(x[i]) for x in raw_data]) for i in range(len(headers))}

    @__display_time('Data processing')
    def process(self, keys=None):
        cdf, rho = {}, {}

        for k, l in [(k, l) for (k, l) in self.data.items() if k in keys] if keys else self.data.items():
            cdf[k], rho[k] = {}, {}

            for n in self.chunk_sizes:
                chunks = (l[i:i+n] for i in range(0, len(l), n))
                cdf[k][n] = [dict(zip(sorted(c), scipy.stats.norm.cdf(sorted(c)))) for c in chunks]
                rho[k][n] = np.empty(len(cdf[k][n]) - 1, dtype=float)

                for i in range(len(cdf[k][n]) - 1):
                    fv, sv, ans = 0, 0, 0
                    for value in sorted(set(cdf[k][n][i]) | set(cdf[k][n][i+1])):
                        if value in cdf[k][n][i]:
                            fv = cdf[k][n][i][value]
                        if value in cdf[k][n][i+1]:
                            sv = cdf[k][n][i+1][value]
                        ans = max(ans, abs(fv - sv))
                    rho[k][n][i] = ans

        self.cdf.update(cdf)
        self.rho.update(rho)
