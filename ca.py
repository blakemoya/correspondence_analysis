import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ca:
    """
    A class for calculating and storing the results of a correspondence analysis of a contingency table,
    based off of the ca library for R

    ...

    Attributes
    __________

    sv : 
    nd : 
    rownames : 
    rowmass : 
    rowdist : 
    rowinertia : 
    rowcoord : 
    rowsup : 
    colnames : 
    colmass : 
    coldist : 
    colinertia : 
    colcoord : 
    colsup : 
    N : 
    """

    def __init__(self, contingency_table):
        """Appendix A of Greenacre 2017"""
        self.rownames = contingency_table.index.values
        self.colnames = contingency_table.columns.values
        self.N = contingency_table.values
        n = contingency_table.sum().sum()
        p = contingency_table.values / n
        self.rowmass = p.sum(axis=1)
        self.colmass = p.sum(axis=0)

         # Step 1: Calculate the matrix S of standardized residuals
        r = self.rowmass
        c = self.colmass
        diag_r = np.diag(r)
        diag_c = np.diag(c)
        diag_r_inv = np.diag(np.reciprocal(r))
        diag_c_inv = np.diag(np.reciprocal(c))
        diag_r_invroot = np.diag(np.reciprocal(np.sqrt(r)))
        diag_c_invroot = np.diag(np.reciprocal(np.sqrt(c)))
        stand_resid = diag_r_invroot @ (p - np.outer(r, c)) @ diag_c_invroot

        # Step 2: SVD of standardized residuals
        u, s, vt = np.linalg.svd(stand_resid, full_matrices=False)
        self.sv = s[:-1]
        out1 = pd.DataFrame([(s ** 2)[:-1], ((s ** 2) / (s ** 2).sum())[:-1]],
                            index = ['Value', 'Percentage'],
                            columns = range(1, len(s)))
        out1.loc['Percentage'] = out1.loc['Percentage'].apply('{:.1%}'.format)
        print(' Principal inertias (eigenvalues):')
        print(out1)
        
        diag_s = np.diag(s)

        # Step 3: Standard coordinates of rows, phi
        phi = diag_r_invroot @ u
        dims = list(map(lambda x: 'Dim. ' + str(x), list(i for i in range(1, phi.shape[1]))))
        self.rowcoord = pd.DataFrame(phi[:, :-1], index = self.rownames, columns = dims)

        # Step 4: Standard coordinates of columns, gamma
        gam = diag_c_invroot @ vt.T
        self.colcoord = pd.DataFrame(gam[:, :-1], index = self.colnames, columns = dims)

        # Step 5 : Principal coordinates of rows, F
        f = phi @ diag_s

        # Step 6: Principal coordinates of columns, G
        g = gam @ diag_s

        # Step 7: Principal intertias //not working
        # how to get chi-square distances? is this hidden somewhere around here?
        '''
        diag_lambda = None
        if len(r) <= len(c):
            diag_lambda = f @ diag_r @ f.T
        else:
            diag_lambda = g @ diag_c @ g.T
        '''
        q_r = diag_r_inv @ p @ diag_c_inv @ p.T @ diag_r_inv
        qq_r = np.array([np.diag(q_r)]).T
        ones = np.ones((len(q_r), 1))
        chi22_r = qq_r @ ones.T - ones @ qq_r.T + 2 * q_r
        print(qq_r)
        print(qq_r @ ones.T)
        print(ones @ qq_r.T)
        print(2 * q_r)
        print(chi22_r)

        self.nd = None
        self.rowdist = None
        self.rowinertia = None
        self.rowsup = None
        self.coldist = None
        self.colinertia = None
        self.colsup = None   


if __name__ == "__main__":
    cont = pd.read_csv('src.csv', index_col=0, header=0)
    C = ca(cont)
    '''
    print(C.sv)
    print(C.nd)
    print(C.rownames)
    print(C.rowmass)
    print(C.rowdist)
    print(C.rowinertia)
    print(C.rowcoord)
    print(C.rowsup)
    print(C.colnames)
    print(C.colmass)
    print(C.coldist)
    print(C.colinertia)
    print(C.colcoord)
    print(C.colsup)
    print(C.N)
    '''