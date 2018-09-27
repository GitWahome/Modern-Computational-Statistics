import math
import scipy.stats as ssstats
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


def pdfFunct(f, x, N):
    """
        math.factorial(N)/(math.factorial(x)
                              *math.factorial(N-x))*(f**x)*((1-f)**(N-x))
        """
    return ssstats.binom.pmf(x, N, f)


def cdfFunction(f, x, N):
    """
    return (math.factorial(N)*(f**x) * (1-f)*(N-x))/\
           (math.factorial(x)*math.factorial(N-x))
    """
    return ssstats.binom.cdf(x, N, f)

def qfFunction(f, x, N):
    """
    return (math.factorial(N)*(f**x) * (1-f)*(N-x))/\
           (math.factorial(x)*math.factorial(N-x))
    """
    return ssstats.binom.ppf(x, N, f)