import os
import subprocess

def calculate_cl_with_xfoil(dat_file, aoa=5.0, reynolds=3e6, xfoil_exec="./xfoil"):
    if not os.path.exists(dat_file):
        print(f"Erreur : le fichier {dat_file} n'existe pas.")
        return None

    # Fichier de sortie
    output_file = "xfoil_output.txt"

    # On envoie un script complet à XFOIL
    xfoil_input = f"""
LOAD naca2412.dat
PANE
OPER
VISC 3e6
ITER 200
ALFA 5
PACC
xfoil_output.txt

QUIT
"""

    try:
        process = subprocess.Popen(
            [xfoil_exec],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=xfoil_input.encode())

        # Si XFOIL s'est arrêté avec un code != 0, on affiche le message d'erreur
        if process.returncode != 0:
            print(f"XFOIL a renvoyé un code de sortie = {process.returncode}")
            print(stderr.decode(errors='ignore'))
            return None

        # On parse xfoil_output.txt pour trouver la ligne contenant "alpha, CL, ..."
        cl_value = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "alpha" in line.lower():
                        # C'est l'en-tête, on l'ignore
                        continue
                    if line.strip() and not line.startswith("-----"):
                        # Normalement, c'est la ligne de résultats
                        # Format: alpha CL CD CDp CM ...
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                # alpha = float(parts[0])  # (optionnel si vous voulez capturer l'alpha)
                                cl_value = float(parts[1]) # CL est la 2ème colonne
                            except:
                                pass
                        break

            # Nettoyage du fichier de sortie
            os.remove(output_file)

        return cl_value

    except Exception as e:
        print(f"Erreur d'exécution XFOIL : {e}")
        return None

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    # Chemin du fichier DAT téléchargé (même dossier)
    dat_file = "naca2412.dat"

    # Chemin vers l'exécutable XFOIL
    # Assurez-vous d'avoir un binaire xfoil nommé "xfoil" dans le dossier courant,
    # ou mettez le chemin complet, ex: xfoil_exec="/usr/local/bin/xfoil"
    xfoil_exec = "./xfoil"

    # Angle d'attaque
    aoa = 5.0
    # Nombre de Reynolds
    reynolds = 3e6

    cl = calculate_cl_with_xfoil(dat_file, aoa=aoa, reynolds=reynolds, xfoil_exec=xfoil_exec)
    if cl is not None:
        print(f"CL for {dat_file} at AoA={aoa}°, Re={reynolds} is: {cl}")
    else:
        print(f"Failed to calculate CL for {dat_file}")
