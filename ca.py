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
        # Return a set with the following values in it
        self.rownames = contingency_table.index.values
        self.colnames = contingency_table.columns.values
        self.N = self.freq_tab(contingency_table)
        self.rowmass = self.N.sum(axis=1)
        self.colmass = self.N.sum(axis=0)
        self.sv = None
        self.nd = None
        self.rowdist = None
        self.rowinertia = None
        self.rowcoord = None
        self.rowsup = None
        self.coldist = None
        self.colinertia = None
        self.colcoord = None
        self.colsup = None

    def analyze(self):
        """Analysis as per Appendix A of Greenacre 2017"""
        # Step 1: Calculate the matrix S of standardized residuals
        prof = self.N.values
        r = self.rowmass.values
        c = self.colmass.values
        diag_r_invroot = np.diag(np.reciprocal(np.sqrt(r)))
        diag_c_invroot = np.diag(np.reciprocal(np.sqrt(c)))
        stand_resid = diag_r_invroot @ (prof - np.outer(r, c)) @ diag_c_invroot
        # Step 2: SVD of standardized residuals
        u, s, vt = np.linalg.svd(stand_resid, full_matrices=False)
        diag_s = np.diag(s)
        # Step 3: Standard coordinates of rows, phi
        phi = diag_r_invroot @ u
        # Step 4: Standard coordinates of columns, gamma
        gam = diag_c_invroot @ vt.T
        # Step 5 : Principal coordinates of rows, F
        f = phi @ diag_s
        # Step 6: Principal coordinates of columns, G
        g = gam @ diag_s

    
    def freq_tab(self, contingency_table):
        n = contingency_table.sum().sum()
        freq = contingency_table / n
        return freq
    


if __name__ == "__main__":
    cont = pd.DataFrame([[6, 1, 11], [1, 3, 11], [4, 25, 0], [2, 2, 20]],
                        columns=['Holidays', 'Half Days', 'Full Days'],
                        index=['Norway', 'Canada', 'Greece', 'France/Germany'])
    C = ca(cont)
    C.analyze()