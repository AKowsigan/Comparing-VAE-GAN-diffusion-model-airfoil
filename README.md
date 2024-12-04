# Comparing-VAE-GAN-diffusion-model-airfoil

## Instalation

```pip install -r requirments.txt```

## Dataset : NACA 4-digit airfoil data

the lift coefficient was calculated for all the four-digit airfoils, and those whose CL could not be calculated or CL < 0 or CL > 2.0 were eliminated. After elimination, the total number of airfoils was 3709.

##  Conditional GAN

```python -m gan.train```

```python -m gan.eval```

```python -m gan.plot```