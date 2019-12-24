import pandas as pd

class ca:
    """
    A class for calculating and storing the results of a correspondence analysis of a contingency table, based off of the ca library for R

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
        self.rownames = contingency_table.index.values
        self.colnames = contingency_table.columns.values
        self.N = self.freq_tab(contingency_table)
        self.rowmass = self.N.sum(axis=0)
        self.colmass = self.N.sum(axis=1)
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
    
    def freq_tab(self, contingency_table):
        n = contingency_table.sum().sum()
        freq = contingency_table / n
        return freq
    


if __name__ == "__main__":
    cont = pd.DataFrame([[0, 10, 0], [5, 5, 5], [30, 5, 0]], columns=['a', 'b', 'c'], index=['x', 'y', 'z'])
    analysis = ca(cont)
    print(analysis.rowmass)
    print(analysis.colmass)
    print(analysis.rowmass.sum())
    print(analysis.colmass.sum())