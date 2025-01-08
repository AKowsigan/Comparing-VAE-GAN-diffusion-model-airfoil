# -*- coding: utf-8 -*-
import numpy as np
from xfoil_utils import polar

def main():

    # afile = "naca2412.dat"
    afile ="wgan_gp/results/xfoil/airfoil_10.dat"
    

    # Set Reynolds number
    re = 3_000_000  

    # Define angles of attack to evaluate
    # angles = [0, 2, 4, 6, 8, 10] 
    angles = [5] 

    # Call the 'polar' function, which internally calls 'calc_polar' + 'read_polar' + 'delete_polar'
    # We'll pass the angles in via the 'alfaseq' argument.
    # Note: 'clseq' is for specifying target CL values (not used here).
    # 'refine=False' means we won't do XFOIL's geometry refinement step.
    # 'max_iter' sets the iteration limit in XFOIL.
    results = polar(
        afile=afile,
        re=re,
        alfaseq=angles,
        refine=False,
        max_iter=50
    )

    # The 'results' dictionary contains arrays: 'a', 'cl', 'cd', 'cdp', 'cm', 'xtr_top', 'xtr_bottom'.
    # Let's print out alpha and CL side by side:
    print(f"Results for airfoil '{afile}' at Re={re}")
    for alpha_val, cl_val, cd_val in zip(results['a'], results['cl'], results['cd']):
        print(f" alpha={alpha_val:>5.2f}  CL={cl_val:>7.4f}  CD={cd_val:>7.5f}")

    # Optionally, you could do something like plotting:
    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(results['a'], results['cl'], 'o-')
    # plt.xlabel('Angle of attack (deg)')
    # plt.ylabel('Lift coefficient, CL')
    # plt.title('Polar curve for ' + afile)
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
