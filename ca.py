import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    def __init__(self,
                 obj,
                 nd=None,
                 suprow=None,
                 supcol=None,
                 subsetrow=None,
                 subsetcol=None):
        """
        Python port of ca.r found here:
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
    
    def plot(self,
             dim=(1, 2),
             map='symmetric',
             what=('all', 'all'),
             mass=(False, False),
             contrib=('none', 'none'),
             col=('blue', 'red'),
             pch=('o', '^'),
             labels=(2, 2),
             arrows=(False, False),
             lines=(False, False),
             lwd=1,
             xlab="_auto_",
             ylab="_auto_",
             col_lab=('blue', 'red')):
        '''
        Plotting ca objects
        '''
        # This is where I'll recycle input if given one value

        # This is where I'll handle supplementary rows/columns for gab and green plotting

        # This is where ca sign switches
        # Principal Coordinates:
        k = len(self.rowcoord.columns)
        rpc = self.rowcoord * self.sv[:k]
        cpc = self.colcoord * self.sv[:k]
        symrpc = self.rowcoord * np.sqrt(self.sv[:k])
        symcpc = self.colcoord * np.sqrt(self.sv[:k])
        # Maptype
        mtlut = {'symmetric': [rpc, cpc],
                 'rowprincipal': [rpc, self.colcoord],
                 'colprincipal': [self.rowcoord, cpc],
                 'symbiplot': [symrpc, symcpc],
                 'rowgab': [rpc, self.colcoord.multiply(self.colmass, axis=0)],
                 'colgab': [self.rowcoord.multiply(self.rowmass, axis=0), cpc],
                 'rowgreen': [rpc, self.colcoord.multiply(np.sqrt(self.colmass), axis=0)],
                 'colgreen': [self.rowcoord.multiply(np.sqrt(self.rowmass), axis=0), cpc]}
        x = mtlut[map][0]
        y = mtlut[map][1]
        # Profiles to plot

        # Dimensions to plot (simple slice in ax.scatter)
        # Build radius/mass vectors

        # Build contribution/color intensity vectors

        # Plot
        ax = plt.axes()
        ax.axvline(ls='--', lw=0.5, c='black')
        ax.axhline(ls='--', lw=0.5, c='black')
        # Set margins

        # Label axes
        ev = np.round(self.sv ** 2, decimals=6)
        pct = np.round(100 * (ev / ev.sum()), decimals=1)
        if xlab == '_auto_':
            ax.set_xlabel(f'Dimension {dim[0]} ({pct[dim[0] - 1]}%)')
        if ylab == '_auto_':
            ax.set_ylabel(f'Dimension {dim[1]} ({pct[dim[1] - 1]}%)')
        # Scatter and annotate
        ax.scatter(x.iloc[:, dim[0] - 1], x.iloc[:, dim[1] - 1], marker=pch[0], c=col[0])
        for a, b, z in zip(x.iloc[:, dim[0] - 1], x.iloc[:, dim[1] - 1], self.rownames):
            label = z
            plt.annotate(label, (a, b), c=col_lab[0])
        ax.scatter(y.iloc[:, dim[0] - 1], y.iloc[:, dim[1] - 1], marker=pch[1], c=col[1])
        for a, b, z in zip(y.iloc[:, dim[0] - 1], y.iloc[:, dim[1] - 1], self.colnames):
            label = z
            plt.annotate(label, (a, b), c=col_lab[1])
        return ax


if __name__ == "__main__":
    print('Testing with src.csv')
    cont = pd.read_csv('src.csv', index_col=0, header=0)
    C = ca(cont)
    print(C)
    for attr, value in C.__dict__.items():
        print(f'\n\t{attr}\n{value}')
    print('Testing plot maptypes')
    for type in ('symmetric', 'rowprincipal', 'colprincipal', 'symbiplot', 'rowgab', 'colgab', 'rowgreen', 'colgreen'):
        C.plot(map=type)
        plt.show()
