import  numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import control.matlab as ml


#numerador = [1, 0, 0, 0, 0]
#denominador = [1, 2463, 3.033e6, 2.118e9, 7.89e11]
numerador = [-1,0, 0]
denominador = [1, 1785.714286, 3188775.51]

roots = np.roots(denominador)
print("cantidad de raices")
print(len(roots))
print("raices:")
print(roots)

print("Reconstruyo para verificar")
coeficientes = poly.polyfromroots(roots)
print("Coeficientes: ")
print(coeficientes)

G = ml.tf(numerador, denominador)
print(G)
print("ceros: ")
print(ml.zero(G))
print("polos: ")
print(ml.pole(G))

def invert(t):
    return 1/t

#mod, fase, w = ml.bode(G)
#wn=ml.damp(G)

yout, T = ml.step(G)
#yout, T = ml.impulse(G)

#x = np.linspace(0, 0.0005, 1000)

#freq = 10000
#signal = np.sin(2*np.pi*freq*x)
#plt.plot(x, np.sin(2*np.pi*freq*x))

#plt.xlabel('Time [seg]')

#plt.ylabel('sin(x)')

#plt.axis('tight')

#yout, T, xout = ml.lsim(G, U = signal, T = x)

#plt.show()

plt.plot(T,yout)
plt.show()

#from scipy import signal

#t = np.linspace(0, .01, 500000, endpoint=False)

#pwm = signal.square(2 * np.pi * 500 * t)

#yout, T, xout = ml.lsim(G, U = pwm, T = t)
#plt.plot(T,yout)
#plt.show()
