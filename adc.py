import  numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import control.matlab as ml
from scipy import signal

def aumentar_x_veces_20dB(x, numerador):
    return (10**x)*numerador

numerador = [-1,0, 0]
denominador = [1, 1785.714286, 3188775.51]

#Transferencia a partir de polos y ceros
ceros = [-100]
polos = [-1000, -10000]


numerador = poly.polyfromroots(ceros)
denominador = poly.polyfromroots(polos)

numerador = numerador[::-1]
denominador = denominador[::-1]

numerador = aumentar_x_veces_20dB(4,numerador)

G = ml.tf(numerador, denominador)
print(G)
print("ceros: ")
print(ml.zero(G))
print("polos: ")
print(ml.pole(G))
ml.damp(G)
#mod, fase, w = ml.bode(G)
#plt.show()

"""
#calcular polos
roots = np.roots(denominador)
print("cantidad de raices")
print(len(roots))
print("raices:")
print(roots)

print("Reconstruyo para verificar")
coeficientes = poly.polyfromroots(roots)
print("Coeficientes: ")
print(coeficientes)

#Analisis rapido de la transferencia
G = ml.tf(numerador, denominador)
print(G)
print("ceros: ")
print(ml.zero(G))
print("polos: ")
print(ml.pole(G))
ml.damp(G)

#Grafico Bode
G = ml.tf(numerador, denominador)
mod, fase, w = ml.bode(G)
plt.show()

#respuesta al escalon
G = ml.tf(numerador, denominador)
yout, T = ml.step(G)
plt.plot(T,yout)
plt.show()

#respuesta al impulso
G = ml.tf(numerador, denominador)
yout, T = ml.impulse(G)
plt.plot(T,yout)
plt.show()
"""
#respuesta a la cuadrada o al seno
G = ml.tf(numerador, denominador)
frecuencia = 500 #en Hz
periodo = 1/frecuencia #en segundos
t = np.linspace(0, 3*periodo, 100000, endpoint=False)

sig = signal.square(2 * np.pi * frecuencia * t)
#sig = np.sin(2*np.pi*frecuencia*t)

yout, T, xout = ml.lsim(G, U = sig, T = t)
plt.plot(T,yout)
#plt.plot(t, sig)
plt.show()