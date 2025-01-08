# xfoil_utils.py

# -*- coding: utf-8 -*-
import os
import numpy as np
import subprocess as sp
import re

def polar(afile, re, *args,**kwargs):
    """calculate airfoil polar and load results
    
    Parameters
    ----------
    afile: string
        path to airfoil dat file or a NACA 4-digit code (e.g. '2412')
    re: float
        fixed reynolds number for polar calculation
    *args, **kwargs
        forwarded to 'calc_polar'
        
    Returns
    -------
    dict
        airfoil polar
    """
    calc_polar(afile, re, 'polar.txt', *args, **kwargs)
    data = read_polar('polar.txt')
    delete_polar('polar.txt')
    return data

def calc_polar(afile, re, polarfile, alfaseq=[], clseq=[], refine=False, max_iter=200, n=None):
    """run xfoil to generate polar file
    
    Parameters
    ----------
    afile: string
        path to airfoil dat file or a NACA 4-digit code
    re: float
        fixed reynolds number
    polarfile: string
        path where the polar data is written
    alfaseq: iterateable, optional
        sequence of angles of attack
    clseq: iterateable, optional
        sequence of lift coefficients (either these or alfaseq must be defined)
    refine: bool
        shall xfoil refine airfoil geometry using GDES?
    max_iter: int
        maximal number of boundary layer iterations
    n: int
        boundary layer parameter (not fully used here)
    """

    if os.name == 'posix':
        xfoilbin = 'xfoil'
    elif os.name == 'nt':
        xfoilbin = 'xfoil.exe'
    else:
        print(f"Operating system {os.name} not supported.")
        return

    pxfoil = sp.Popen([xfoilbin], stdin=sp.PIPE, stdout=None, stderr=None)

    def write2xfoil(string):
        # For Python 3, encode the string
        pxfoil.stdin.write(string.encode('ascii'))

    # If you gave a string like "2412", treat as built-in NACA
    if afile.isdigit():
        write2xfoil(f"NACA {afile}\n")
    else:
        # Otherwise treat as a .dat file
        write2xfoil(f"LOAD {afile}\n")
        
        if refine:
            # Example geometry editing
            write2xfoil("GDES\n")
            write2xfoil("CADD\n")
            write2xfoil("\n\n\n")
            write2xfoil("X\n ")
            write2xfoil("\n")
            write2xfoil("PANEL\n")

    write2xfoil("OPER\n")
    # if n is not None:  # you had a "if False: %n != None" in the original snippet, might have been a typo
    #     write2xfoil("VPAR\n")
    #     write2xfoil(f"N {n}\n\n")

    write2xfoil(f"ITER {max_iter}\n")
    write2xfoil("VISC\n")
    write2xfoil(f"{re}\n")
    write2xfoil("PACC\n")
    write2xfoil("\n")        # blank line for polar file (not used)
    write2xfoil("\n")        # second blank line
    for alfa in alfaseq:
        write2xfoil(f"A {alfa}\n")
    for cl in clseq:
        write2xfoil(f"CL {cl}\n")
    write2xfoil("PWRT 1\n")
    write2xfoil(f"{polarfile}\n")
    write2xfoil("\n")

    pxfoil.communicate("quit".encode('ascii'))

def read_polar(infile):
    """read xfoil polar results from file
    
    Returns
    -------
    dict
        dictionary with polar data
    """
    regex = re.compile(r'(?:\s*([+-]?\d*\.?\d+))')
    
    with open(infile) as f:
        lines = f.readlines()
        
    a, cl, cd, cdp, cm = [], [], [], [], []
    xtr_top, xtr_bottom = [], []
    
    # Usually first 12 lines are header comments in XFOIL polar
    for line in lines[12:]:
        linedata = regex.findall(line)
        # linedata should contain something like [alpha, CL, CD, CDp, CM, xtr_top, xtr_bottom]
        if len(linedata) >= 7:
            a.append(float(linedata[0]))
            cl.append(float(linedata[1]))
            cd.append(float(linedata[2]))
            cdp.append(float(linedata[3]))
            cm.append(float(linedata[4]))
            xtr_top.append(float(linedata[5]))
            xtr_bottom.append(float(linedata[6]))

    data = {
        'a': np.array(a),
        'cl': np.array(cl),
        'cd': np.array(cd),
        'cdp': np.array(cdp),
        'cm': np.array(cm),
        'xtr_top': np.array(xtr_top),
        'xtr_bottom': np.array(xtr_bottom)
    }
    return data

def delete_polar(infile):
    """ deletes polar file """
    if os.path.exists(infile):
        os.remove(infile)
