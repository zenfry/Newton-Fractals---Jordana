from newton import newton
import numpy as np
import matplotlib.pyplot as plt

def test_cubic(z):
    return z**3 - 0.00004

def test_cubic_prime(z):
    return 3*z**2

def main():

    xs = np.linspace(-2, 2, 700)
    ys = np.linspace(-2, 2, 700)
    zs = xs[:, None] + 1j * ys  

    
    roots = newton(test_cubic, zs, fprime=test_cubic_prime)

    
    unique_roots = np.unique(np.round(roots, decimals=6))
    root_map = {root: i for i, root in enumerate(unique_roots)}
    colors = np.vectorize(lambda r: root_map.get(np.round(r, decimals=6), -1))(roots)

    plt.figure(figsize=(6, 6))
    plt.imshow(colors, extent=(-2.5, 2.5, -2.5, 2.5), origin='lower', cmap='hsv')
    plt.title("Newton Fractal 1")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    main()
