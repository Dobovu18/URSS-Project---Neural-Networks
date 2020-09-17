import numpy as np
#Activation functions from https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/
#and https://en.wikipedia.org/wiki/Activation_function
    
    #Activation functions are:
    #    1.  Identity
    #    2.  Step
    #    3.  Sigmoid
    #    4.  Tanh
    #    5.  Arctan
    #    6.  Softsign
    #    7.  ISRU
    #    8.  ISRLU
    #    9.  SQNL
    #    10. ReLU
    #    11. LeakyReLU
    #    12. ELU
    #    13. Softplus
    #    14. BentID
    #    15. Swish
    #    16. Ladle
    


def Identity(x):
    #Works well, 82%
    return x
    
def d_Identity(x):
    return 1

def Step(x):
    #Works, but poorly, as expected.
    return np.heaviside(x,1)

def d_Step(x):
    return 0

def Sigmoid(x):
    #Great
    return 1.0 / (1.0 + np.exp(-x))

def d_Sigmoid(x):
    #Great
    return Sigmoid(x) * (1 - Sigmoid(x))

def Tanh(x):
    #Great
    return np.tanh(x)

def d_Tanh(x):
    return 1.0 - Tanh(x)**2

def Arctan(x):
    #Great
    return np.arctan(x)

def d_Arctan(x):
    return 1.0 / (1 + x**2)

def Softsign(x):
    #results!
    return x / (1 + abs(x))

def d_Softsign(x):
    #results!
    return 1.0 / (1.0 + abs(x))**2

def ISRU(x, a = 0.01):
    #Meh!
    return x / np.sqrt(1 + a*x**2)

def d_ISRU(x, a = 0.01):
    return (1 / np.sqrt(1+ a*x**2))**3

def ISRLU(x, a = 0.01):
    #tested!
    ans = np.piecewise(x,
                        [x < 0, x >= 0],
                        [lambda x: x / np.sqrt(1 + a*x**2), lambda x: x])
    return ans

def d_ISRLU(x, a = 0.01):
    ans = np.piecewise(x,
                        [x < 0, x >= 0],
                        [lambda x: np.sqrt(1 + a*x**2)**-3, 1])
    return ans

def SQNL(x):
    #very good
    ans = np.piecewise(x, 
                        [x > 0, x <= 0, x <= -2, x >= 2],
                        [lambda x: x - 0.25*x**2.0, lambda x: x + 0.25*x**2.0, -1, 1])
    return ans

def d_SQNL(x):
    ans = np.piecewise(x, 
                        [abs(x) <= 2.0, abs(x) > 2.0],
                        [lambda x: 1 - 0.5 * abs(x), 0])
    return ans

def ReLU(x):
    #finally able to get some weird results
    return np.maximum(0, x)

def d_ReLU(x):
    ans = np.piecewise(x, 
                        [x < 0, x >= 0], 
                        [0, 1])
    return ans

    
def LeakyReLU(x, a = 0.01):
    #working great, better than ReLU imo
    return np.maximum(a*x, x)

def d_LeakyReLU(x, a = 0.01):
    ans = np.piecewise(x, 
                        [x < 0, x >= 0], 
                        [a, 1])
    return ans

def ELU(x, a = 0.01):
    #good
    ans = np.piecewise(x, 
                        [x < 0, x >= 0], 
                        [lambda x: a * (np.exp(x) - 1), lambda x: x])
    return ans

def d_ELU(x, a = 0.01):
    ans = np.piecewise(x, [x <= 0, x > 0], [lambda x: ELU(x, a) + a, 1])
    return ans

def Softplus(x):
    #Very good as well
    return np.log(1 + np.exp(x))

def d_Softplus(x):
    return Sigmoid(x)

def BentId(x):
    #good
    return 0.5 * (np.sqrt(x**2 + 1) - 1) + x
    
def d_BendId(x):
    return 0.5 * (x / np.sqrt(x**2 + 1)) + 1

def Swish(x):
    #https://arxiv.org/abs/1710.05941
    #poor performance :(
    return x * Sigmoid(x)
    
def d_Swish(x):
    return Swish(x) + Sigmoid(x) * (1 - Swish(x))

def Ladle(x, a = 0.01):
    #An activation function I made up, work well
    ans = np.piecewise(x, 
                    [x <= 0, x > 0], 
                    [lambda x: x * np.exp(a * x), lambda x: x])
    return ans

def d_Ladle(x, a = 0.01):
    ans = np.piecewise(x, 
                    [x <= 0, x > 0], 
                    [lambda x: (a*x + 1) * np.exp(a*x), lambda x: 1])
    return ans