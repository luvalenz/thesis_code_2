import numpy as np
import pandas as pd
import glob
import pickle
import sys
import os

# def self_product(x):
#     return np.matrix(x).T*np.matrix(x)


# def cov_inc(new_row, x_mean, x_cov, x_std, n):
#     new_row = np.matrix(new_row)
#     xtx_ = IncrementalPCA.xtx(x_cov, x_std, x_mean, n)
#     xtx_inc = xtx_ + new_row.T*new_row
#     d = np.matrix(np.diag(xtx_))
#     std_inc = np.sqrt(IncrementalPCA.var_inc(d, new_row, x_mean, n))
#     a = std_inc.T*std_inc
#     a_inv = 1/a
#     x_inc_mean = (n*x_mean + new_row)/(n+1)
#     b = xtx_inc - (n+1)*x_inc_mean.T*x_inc_mean
#     return np.multiply(a_inv,b)/(n+1)


class IncrementalPCA:
    def __init__(self, n_components = None):
        self.n_components = n_components
        self.cov = None
        self._W = None

    @staticmethod
    def standarize(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return np.nan_to_num((X - mean)/std)

    @staticmethod
    def cov(X):
        X_std = IncrementalPCA.standarize(X)
        n = len(X)
        return np.nan_to_num(np.matrix(X_std).T*np.matrix(X_std))/n

    @staticmethod
    def xtx(cov, std, mean, n):
        a = std.T*std
        b = mean.T*mean
        xtx_ = n*np.multiply(a,cov)
        return xtx_ + n*b

    @staticmethod
    def var_inc(d, new_row, x_mean, n):
        new_row_sq = np.matrix(np.array(new_row)**2)
        return (d + new_row_sq)/(n + 1) - np.matrix(np.array((n*x_mean + new_row)/(n+1))**2)

    @staticmethod
    def cov_stack(x2, x1_mean, x1_cov, x1_std, n1):
        xtx1 = IncrementalPCA.xtx(x1_cov, x1_std, x1_mean, n1)
        xtx2 = x2.T*x2
        n2 = len(x2)
        n = n1 + n2
        x2_mean = np.mean(x2, axis=0)
        xtx_stack = xtx1 + xtx2
        d1 = np.matrix(np.diag(xtx1))
        std_stack = np.sqrt(IncrementalPCA.var_stack(d1, x2, x1_mean, n1))
        a = std_stack.T*std_stack
        x_stack_mean = (n1*x1_mean + n2*x2_mean)/n
        b = xtx_stack - n*x_stack_mean.T*x_stack_mean
        return np.nan_to_num(np.true_divide(b,a))/n, x_stack_mean, std_stack

    @staticmethod
    def var_stack(d1, x2, x1_mean, n1):
        d2 = np.diag(x2.T*x2)
        n2 = len(x2)
        x2_mean = np.mean(x2, axis=0)

        d1 = np.float128(d1)
        x1_mean = np.float128(x1_mean)
        n1 = np.float128(n1)
        d2 = np.float128(d2)
        x2_mean = np.float128(x2_mean)
        n2 = np.float128(n2)
        return np.float32((d1 + d2)/(n1 + n2) - np.matrix(np.array((n1*x1_mean + n2*x2_mean)/(n1+n2))**2))

    @property
    def W(self):
        if self._W is None:
            self.calculate_W()
        return self._W

    def add(self, X):
        X = np.matrix(X)
        self._W = None
        if self.cov is None:
            self.cov = IncrementalPCA.cov(X)
            self.mean = np.mean(X, axis= 0)
            self.std = np.std(X, axis=0)
            self.n = len(X)
        else:
            self.cov, self.mean, self.std = IncrementalPCA.cov_stack(X, self.mean, self.cov, self.std, self.n)
            if np.isnan(np.sum(self.cov)):
                print(self.cov)
            n = self.n + len(X)

    def calculate_W(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)
        eigenvalues_order = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[eigenvalues_order]
        sorted_eigenvectors = eigenvectors[:,eigenvalues_order]
        self._W = sorted_eigenvectors
        if self.n_components is not None:
            self._W = self._W[:,:self.n_components]

    def transform(self, X):
        standarized_X = IncrementalPCA.standarize(X)
        return np.dot(standarized_X, self.W)

    def to_pickle(self, name, path):
        output = open(os.path.join(path, '{0}_ipca.pkl'.format(name)), 'wb')
        pickle.dump(self, output)
        output.close()

    @staticmethod
    def from_pickle(path):
        pkl_file = open(path, 'rb')
        return pickle.load(pkl_file)


def compute_field_pca(root, field, ipca):
    paths = glob.glob(root + '/F_{0}_*.csv'.format(field))
    tiles = [path[path.rfind('_')+1:path.rfind('.')] for path in paths]
    for tile in tiles:
        compute_tile_pca(root, field, tile, ipca)
    pickle.dump(ipca, open( "ipca_upto_F_{0}.p".format(field), "wb" ) )

def compute_tile_pca(root, field, tile, ipca):
    path = '{0}/F_{1}_{2}.csv'.format(root, field, tile)
    tile_df = pd.read_csv(path, header=0, index_col=0)
    values = tile_df.values
    ipca.add(values)

def transform_field_pca(root, field, output_path, ipca):
    paths = glob.glob(root + '/F_{0}_*.csv'.format(field))
    tiles = [path[path.rfind('_')+1:path.rfind('.')] for path in paths]
    for tile in tiles:
        transform_tile_pca(root, field, tile, ipca, output_path)
        print("field {0} transformed".format(field))


def transform_tile_pca(root, field, tile, ipca, output_path):
    path = '{0}/F_{1}_{2}.csv'.format(root, field, tile)
    tile_df = pd.read_csv(path, header=0, index_col=0)
    values = tile_df.values
    indices = tile_df.index.values
    pca_values = ipca.transform(values)
    pca_values_df = pd.DataFrame(pca_values, index=indices)
    pca_values_df.to_csv('{0}/F_{1}_{2}.csv'.format(output_path, field, tile), sep=',')


def compute_upto(last_field, root):
    ipca = IncrementalPCA(5)
    for field in range(1, last_field + 1):
        compute_field_pca(root, field, ipca)
        print("field {0} computed".format(field))


def transform_upto(last_field, root, output_path):
    ipca = pickle.load( open( "ipca_upto_F_{0}.p".format(last_field), "rb" ) )
    for field in range(1, last_field + 1):
        transform_field_pca(root, field, output_path, ipca)


if __name__ == '__main__':
    root_path = sys.argv[1]
    output_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/macho_features_pca'
    #compute_upto(82, root_path)
    transform_upto(82, root_path, output_path)
