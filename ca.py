import numpy as np
import pandas as pd


class ca:
    """
    A class for calculating and storing the results of a correspondence
    analysis of a contingency table, based off of the ca library for R.

    ...

    Parameters
    __________
    obj : pandas.DataFrame
        Source two-way frequency table.
    nd : int //not yet implemented
        Number of dimensions to be included in the output; if None the maximum
        number possible dimensions are included.
    suprow : //not yet implemented
        Indices of supplementary rows.
    supcol : //not yet implemented
        Indices of supplementary columns.
    subsetrow : //not yet implemented

    subsetcol : //not yet implemented


    Attributes
    __________
    sv : numpy.ndarray
        Singular values
    nd : int or NoneType
        Dimension of the solution
    rownames : numpy.ndarray
        Row names
    rowmass : numpy.ndarray
        Row masses
    rowdist : numpy.ndarray
        Row chi-square distances to centroid
    rowinertia : numpy.ndarray
        Row inertias
    rowcoord : pandas.DataFrame
        Row standard coordinates
    rowsup : //not yet implemented
        Indices of supplementary row points
    colnames : numpy.ndarray
        Column names
    colmass : numpy.ndarray
        Column masses
    coldist : numpy.ndarray
        Column chi-square distances to centroid
    colinertia : numpy.ndarray
        Column inertias
    colcoord : pandas.DataFrame
        Column standard coordinates
    colsup : //not yet implemented
        Indices of supplementary columns
    N : numpy.ndarray
        The frequency table
    """

    def __init__(self, obj, nd=None, suprow=None, supcol=None,
                 subsetrow=None, subsetcol=None):
        """
        Python translation of ca.r found here:
        https://r-forge.r-project.org/scm/viewvc.php/?root=ca0
        """
        self.nd = nd
        i = len(obj.index)
        j = len(obj.columns)
        self.rownames = obj.index.values
        self.colnames = obj.columns.values
        self.N = obj.values
        # This is where I'll handle supplementary rows/columns

        # This is where I'll adjust for subset CA

        # Init:
        n = obj.sum().sum()
        p = self.N / n
        self.rowmass = p.sum(axis=1)
        self.colmass = p.sum(axis=0)
        # SVD:
        expected_p = np.outer(self.rowmass, self.colmass)
        expected_N = expected_p * n
        s = (p - expected_p) / np.sqrt(expected_p)
        # This is where I'll do subset CA

        chimat = s ** 2 * n
        u, sv, vt = np.linalg.svd(s, full_matrices=False)
        self.sv = sv[:-1]  # This should later become [:nd.max]
        ev = self.sv ** 2
        cumev = np.cumsum(ev)
        # Intertia:
        totin = ev.sum()
        self.rowinertia = (s ** 2).sum(axis=1)
        self.colinertia = (s ** 2).sum(axis=0)
        # chidist
        self.rowdist = np.sqrt(self.rowinertia / self.rowmass)
        self.coldist = np.sqrt(self.colinertia / self.colmass)
        # This is where I'll handle subset CA and supplementary row/columns

        # Standard coordinates:
        phi = np.divide(u[:, :-1], np.sqrt(self.rowmass)[np.newaxis, :].T)
        # [:, :nd.max ]
        gam = np.divide(vt.T[:, :-1], np.sqrt(self.colmass)[np.newaxis, :].T)
        # [:, :nd.max ]
        # This is where I'll handle standard coordinates for supplementary
        # rows/columns

        dims = list(map(lambda x: 'Dim. ' + str(x),
                    list(i for i in range(1, phi.shape[1] + 1))))
        self.rowcoord = pd.DataFrame(phi, index=self.rownames, columns=dims)
        self.colcoord = pd.DataFrame(gam, index=self.colnames, columns=dims)
        self.rowsup = None
        self.colsup = None
        if self.nd == None:
            self.nd = len(self.sv)

    def __str__(self):
        """
        Printing ca objects
        """
        # Eigenvalues:
        value = np.round(self.sv ** 2, decimals=6)
        percentage = value / value.sum()
        eigenvalues = pd.DataFrame([value, percentage],
                                   index=['Value', 'Percentage'],
                                   columns=range(1, self.nd + 1))
        eigenvalues.loc['Percentage'] = (eigenvalues.loc['Percentage']
                                         .apply('{:.1%}'.format))
        out1 = f' Principal inertias (eigenvalues):\n{eigenvalues}'
        # Row Profiles:
        tmp_values = np.hstack((self.rowmass[:, np.newaxis],
                                self.rowdist[:, np.newaxis],
                                self.rowinertia[:, np.newaxis],
                                self.rowcoord)).T
        # This is where I'll handle supplementary rows

        dims = list(map(lambda x: 'Dim. ' + str(x),
                    list(i for i in range(1, self.nd + 1))))
        row_profiles = pd.DataFrame(tmp_values,
                                    index=(['Mass', 'ChiDist', 'Inertia'] +
                                           dims),
                                    columns=self.rownames)
        out2 = f' Rows:\n{row_profiles}'
        # Column Profiles:
        tmp_values = np.hstack((self.colmass[:, np.newaxis],
                                self.coldist[:, np.newaxis],
                                self.colinertia[:, np.newaxis],
                                self.colcoord)).T
        # This is where I'll handle supplementary columns
        col_profiles = pd.DataFrame(tmp_values,
                                    index=(['Mass', 'ChiDist', 'Inertia'] +
                                           dims),
                                    columns=self.colnames)
        out3 = f' Columns:\n{col_profiles}'
        return(f'\n{out1}\n\n{out2}\n\n{out3}')


if __name__ == "__main__":
    print('Testing with src.csv')
    cont = pd.read_csv('src.csv', index_col=0, header=0)
    C = ca(cont)
    print(C)
    for attr, value in C.__dict__.items():
        print(f'\n\t{attr}\n{value}')
