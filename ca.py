import os
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
    suprow : 
        A list of names of supplementary rows.
    supcol : 
        A list of names of supplementary columns.
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
    rowsup :
        Names of supplementary row points
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
        nd0 = nd
        i = len(obj.index)
        j = len(obj.columns)
        rn = obj.index.values
        cn = obj.columns.values
        N = obj # .values
        # Temporarily remove supplementary rows/columns
        Ntemp = N
        NtempC = N
        NtempR = N
        # Greenacre sorts suprow/col. Don't think I need to bc pandas
        if supcol is not None and suprow is not None:
            NtempC = Ntemp.drop(suprow, axis=0)
            NtempR = Ntemp.drop(supcol, axis=1)
        if supcol is not None:
            sc = NtempC[supcol]
            Ntemp = Ntemp.drop(supcol, axis=1)
            cs_sum = sc.sum(axis=0)
        if suprow is not None:
            sr = Ntemp.loc[suprow]
            Ntemp = Ntemp.drop(suprow, axis=0)
            rs_sum = sr.sum(axis=1)
        N = Ntemp
        # Adjust for subset CA
        # Skipping all index adjustments for now

        # Check for subset CA
        dim_N = N.shape
        if subsetrow is not None:
            if supcol is not None:
                sc = sc.loc[subsetrow]
            if subsetcol is not None:
                sr = sr[subsetcol]
        # End subset CA
        if subsetrow is None and subsetcol is None:
            nd_max = min(N.shape) - 1
        else:
            N00 = N
            if subsetrow is not None:
                N00 = N00.loc[subsetrow]
            if subsetcol is not None:
                N00 = N00[subsetcol]
            dim_N = N00.shape
            nd_max = min(N.shape)
            if subsetrow is not None and subsetcol is None:
                if (dim_N[0] > dim_N[1]):
                    nd_max = min(dim_N) - 1
                elif subsetrow is None and subsetcol is not None:
                    if dim_N[1] > dim_N[0]:
                        nd_max = min(dim_N) - 1
        if nd is None or nd > nd_max:
            nd = nd_max
        # Init:
        n = N.sum().sum()
        p = N / n
        rm = p.sum(axis=1) # is this reversed between numpy and pandas?
        cm = p.sum(axis=0)
        # SVD:
        expected_p = np.outer(rm, cm)
        expected_N = expected_p * n
        S = (p - expected_p) / np.sqrt(expected_p)
        # Subset CA
        if subsetcol is not None:
            S = S[subsetcol]
            cm = cm[subsetcol]
            cn = cn[subsetcol] # Originally subsetcolt
        if subsetrow is not None:
            S = S.loc[subsetrow]
            rm = rm[subsetrow]
            rn = rn[subsetrow] # Originally subsetrowt
        # End sCA
        chimat = S ** 2 * n
        u, sv, vt = np.linalg.svd(S, full_matrices=False)
        sv = sv[:nd_max]
        ev = sv ** 2
        cumev = np.cumsum(ev)
        # Intertia:
        totin = ev.sum()
        rin = (S ** 2).sum(axis=1)
        cin = (S ** 2).sum(axis=0)
        # Chidist
        rachidist = np.sqrt(rin / rm)
        cachidist = np.sqrt(cin / cm)
        rchidist = rachidist # Originally nans(i)
        cchidist = cachidist # Originally nans(j)
        if subsetrow is not None:
            obj = obj.loc[subsetrow] # Originally subsetrowt
        if subsetcol is not None:
            obj = obj[subsetcol] # Originally subsetcolt
        # Handle supplementary row/columns
        if suprow is not None:
            if supcol is None:
                P_stemp = obj.loc[suprow]
            else:
                P_stemp = obj.loc[suprow, ~obj.columns.isin(supcol)]
            P_stemp = P_stemp.divide(P_stemp.sum(axis=1), axis=0)
            P_stemp = (P_stemp - cm) / np.sqrt(cm)
            rschidist = np.sqrt((P_stemp ** 2).sum(axis=1))
            rchidist = rchidist.append(rschidist)
        if supcol is not None:
            if suprow is None:
                P_stemp = obj[supcol]
            else:
                P_stemp = obj.loc[~obj.index.isin(suprow), supcol]
            P_stemp = P_stemp.divide(P_stemp.sum(axis=0), axis=1)
            P_stemp = ((P_stemp.T - rm) / np.sqrt(rm)).T
            cschidist = np.sqrt((P_stemp ** 2).sum(axis=0))
            cchidist = cchidist.append(cschidist)
        # Standard coordinates:
        phi = np.divide(u[:, :nd], np.sqrt(rm)[np.newaxis, :].T)
        phi = pd.DataFrame(phi, index=rn[~np.isin(rn, suprow)])
        gam = np.divide(vt.T[:, :nd], np.sqrt(cm)[np.newaxis, :].T)
        gam = pd.DataFrame(gam, index=cn[~np.isin(cn, supcol)])
        # Standard coordinates for supplementary rows/columns
        if suprow is not None:
            base = sr.divide(rs_sum, axis=0) - cm
            phi_s = (base @ gam).divide(sv[:nd], axis=1)
            phi_s = phi.append(phi_s)
            rm_old = rm
            rm = rm.append(pd.Series(np.empty(len(suprow)), index=suprow))
            rin = rin.append(pd.Series(np.empty(len(suprow)), index=suprow))
        if supcol is not None:
            if suprow is not None:
                base = sc.divide(cs_sum, axis=1).T - rm_old
            else:
                base = sc.divide(cs_sum, axis=1).T - rm
            gam_c = (base @ phi).divide(sv[:nd], axis=1)
            gam = gam.append(gam_c)
            cm = cm.append(pd.Series(np.empty(len(supcol)), index=supcol))
            cin = cin.append(pd.Series(np.empty(len(supcol)), index=supcol))
        if 'phi_s' in locals() or 'phi_s' in globals():
            phi = phi_s
        self.sv = sv
        self.nd = nd
        self.rownames = rn
        self.rowmass = rm
        self.rowdist = rchidist
        self.rowinertia = rin
        self.rowcoord = phi.rename(columns=lambda c: 'Dim. ' + str(c + 1))
        self.rowsup = suprow
        self.colnames = cn
        self.colmass = cm
        self.coldist = cchidist
        self.colinertia = cin
        self.colcoord = gam.rename(columns=lambda c: 'Dim. ' + str(c + 1))
        self.colsup = supcol
        self.N = N

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
                                         .apply('{:.2%}'.format))
        out1 = f' Principal inertias (eigenvalues):\n{eigenvalues}'
        row_profiles = pd.concat([self.rowmass,
                                  self.rowdist,
                                  self.rowinertia,
                                  self.rowcoord],
                                  axis=1).T
        row_profiles = row_profiles.rename(index={0: 'Mass', 1: 'ChiDist', 2: 'Inertia'})
        out2 = f' Rows:\n{row_profiles}'
        # Column Profiles:
        col_profiles = pd.concat([self.colmass,
                                  self.coldist,
                                  self.colinertia,
                                  self.colcoord],
                                  axis=1).T
        col_profiles = col_profiles.rename(index={0: 'Mass', 1: 'ChiDist', 2: 'Inertia'})
        out3 = f' Columns:\n{col_profiles}'
        return(f'\n{out1}\n\n{out2}\n\n{out3}')

    def plot(self,
             ax=None,
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

        # Original here if `if (!is.numeric(x$suprow))`, what does that do?
        '''if self.suprow is None:
            if map == 'colgab' or map == 'colgreen':
                if what[0] is not 'none':
                    what[0] = 'active'
        if self.supcol is None:
            if map == 'rowgab' or map == 'rowgreen':
                if what[1] is not 'none':
                    what[1] = 'active'''
        # This is where ca sign switches
        # Principal Coordinates:
        k = len(self.rowcoord.columns)
        rpc = self.rowcoord * self.sv[:k]
        cpc = self.colcoord * self.sv[:k]
        symrpc = self.rowcoord * np.sqrt(self.sv[:k])
        symcpc = self.colcoord * np.sqrt(self.sv[:k])
        # Maptype
        mt = {'symmetric': [rpc, cpc],
              'rowprincipal': [rpc, self.colcoord],
              'colprincipal': [self.rowcoord, cpc],
              'symbiplot': [symrpc, symcpc],
              'rowgab': [rpc, self.colcoord.multiply(self.colmass, axis=0)],
              'colgab': [self.rowcoord.multiply(self.rowmass, axis=0), cpc],
              'rowgreen': [rpc, self.colcoord.multiply(np.sqrt(self.colmass),
                                                       axis=0)],
              'colgreen': [self.rowcoord.multiply(np.sqrt(self.rowmass),
                                                  axis=0), cpc]}
        x = mt[map][0].copy(deep=True)
        y = mt[map][1].copy(deep=True)
        if self.rowsup is not None:
            x['sup'] = np.where(x.index.isin(self.rowsup), 'none', col[0])
        else:
            x['sup'] = col[0]
        if self.colsup is not None:
            y['sup'] = np.where(y.index.isin(self.colsup), 'none', col[1])
        else:
            y['sup'] = col[1]
        # Profiles to plot

        # Dimensions to plot (simple slice in ax.scatter)
        # Build radius/mass vectors

        # Build contribution/color intensity vectors

        # Plot
        if ax is None:
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
        ax.scatter(x.iloc[:, dim[0] - 1], x.iloc[:, dim[1] - 1],
                   marker=pch[0],
                   edgecolors=col[0],
                   facecolors=x['sup'].values)
        for a, b, z in zip(x.iloc[:, dim[0] - 1], x.iloc[:, dim[1] - 1],
                           self.rowcoord.index.values):
            label = z
            plt.annotate(label, (a, b), color=col_lab[0])
        ax.scatter(y.iloc[:, dim[0] - 1], y.iloc[:, dim[1] - 1],
                   marker=pch[1],
                   edgecolors=col[1],
                   facecolors=y['sup'].values)
        for a, b, z in zip(y.iloc[:, dim[0] - 1], y.iloc[:, dim[1] - 1],
                           self.colcoord.index.values):
            label = z
            plt.annotate(label, (a, b), color=col_lab[1])
        return ax

def basic_test():
    for example in os.listdir('data'):
        print(f'Testing with {example}')
        cont = pd.read_csv(f'data/{example}', index_col=0, header=0)
        C = ca(cont)
        print(C)
        for attr, value in C.__dict__.items():
            print(f'\n\t{attr}\n{value}')
        print(f'Testing plot maptypes on {example}')
        fig = plt.figure()
        for n, type in enumerate(['symmetric', 'rowprincipal', 'colprincipal',
                                  'symbiplot', 'rowgab', 'colgab', 'rowgreen',
                                  'colgreen']):
            ax = fig.add_subplot(2, 4, n + 1)
            ax.set_title(type)
            C.plot(ax=ax, map=type)
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            print('Unable to maximize window')
        plt.show()

def suppl_test():
    for example in os.listdir('data')[0::2]:
        print(f'Testing suprow with {example}')
        cont = pd.read_csv(f'data/{example}', index_col=0, header=0)
        supr = cont.index.values[0:1]
        supc = cont.columns.values[0:1]
        print(cont)
        C = ca(cont, suprow=supr, supcol=supc)
        print(C)
        for attr, value in C.__dict__.items():
            print(f'\n\t{attr}\n{value}')
        print(f'Testing plot maptypes on {example}')
        fig = plt.figure()
        for n, type in enumerate(['symmetric', 'rowprincipal', 'colprincipal',
                                  'symbiplot', 'rowgab', 'colgab', 'rowgreen',
                                  'colgreen']):
            ax = fig.add_subplot(2, 4, n + 1)
            ax.set_title(type)
            C.plot(ax=ax, map=type)
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            print('Unable to maximize window')
        plt.show()

if __name__ == "__main__":
    basic_test()
