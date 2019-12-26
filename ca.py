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
        diag_r_invroot = np.diag(np.reciprocal(np.sqrt(r)))
        diag_c_invroot = np.diag(np.reciprocal(np.sqrt(c)))
        stand_resid = diag_r_invroot @ (p - np.outer(r, c)) @ diag_c_invroot

        # Step 2: SVD of standardized residuals
        u, s, vt = np.linalg.svd(stand_resid, full_matrices=False)
        self.sv = s[:-1]
        principal_inertias = self.sv ** 2
        diag_s = np.diag(s)

        # Step 3: Standard coordinates of rows, phi
        phi = diag_r_invroot @ u

        # Step 4: Standard coordinates of columns, gamma
        gam = diag_c_invroot @ vt.T

        # Step 5 : Principal coordinates of rows, F
        f = phi @ diag_s

        # Step 6: Principal coordinates of columns, G
        g = gam @ diag_s

        # Step 7: Principal intertias
        # how to get chi-square distances? is this hidden somewhere around here?
        diag_lambda = None
        if len(r) <= len(c):
            diag_lambda = f @ diag_r @ f.T
        else:
            diag_lambda = g @ diag_c @ g.T
        print(diag_lambda)

        self.nd = None
        self.rowdist = None
        self.rowinertia = None
        self.rowcoord = None
        self.rowsup = None
        self.coldist = None
        self.colinertia = None
        self.colcoord = None
        self.colsup = None   


if __name__ == "__main__":
    cont = pd.read_csv('src.csv', index_col=0, header=0)
    C = ca(cont)