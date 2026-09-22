import numpy as np

def intensity_closure(intensity, covector, b):
    output = 0
    for i in range(len(covector)):
        cv = covector[i]
        idif = (intensity[cv[1]] - intensity[cv[0]])
        output += cv[-1] * np.arctan(b * idif)    
    return output

def arctan_jac(intensity, covector, b):
    output = 0
    for i in range(len(covector)):
        cv = covector[i]
        idif = (intensity[cv[1]] - intensity[cv[0]])
        output += cv[-1] * idif / (idif**2 * b**2 + 1)
    return output

def forwardClosurePhases(intensity, b, covectors):
    predicted = np.zeros(len(covectors), dtype=np.float32)
    for i in range(len(covectors)):
        predicted[i] = intensity_closure(intensity, covectors[i], b=b)
    return np.exp(1j * predicted)

def jacobian(x, _, intensity, covectors):
    J = np.ones((len(covectors)), dtype=np.float64)
    for i in range(len(covectors)):
        J[i] = arctan_jac(intensity, covectors[i], b=x[0])
    return J[:, np.newaxis]

def closureResiduals(x, observedClosurePhases, intensity, validCovectors):
    predicted = forwardClosurePhases(intensity, x[0], validCovectors)
    residual = np.angle(predicted * observedClosurePhases.conj())
    return residual