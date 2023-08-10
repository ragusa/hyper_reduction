import numpy as np

def energy_content(sig):
    # Compute energy content:
    es = np.zeros(len(sig))
    cum = 0.0 # cumulative e content
    for i in range(len(sig)):
        cum += sig[i]
        es[i] = cum
    es /= es[-1] # Normalize to 1
    return es
