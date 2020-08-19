import  numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import control.matlab as ml
from scipy import signal

numerador = [1, 0, 0, 0, 0] # igual para ambas transferencias
denominador = [1, 2463, 3.033e6, 2.118e9, 7.89e11] #transferencia enunciado

#denominador = [1, 2381, 2.958e6, 1.987e9, 7.404e11] #transferencia encontrada
"""
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

#respuesta a la cuadrada o al seno
G = ml.tf(numerador, denominador)
frecuencia = .8 #en Hz
periodo = 1/frecuencia #en segundos
t = np.linspace(0, 5*periodo, 1000000, endpoint=False)

sig = signal.square(2 * np.pi * frecuencia * t)
#sig = np.sin(2*np.pi*frecuencia*t)
#sig = (sig +1)*0.5#para agregarle un offset a la señal cuadrada de valor .5v
yout, T, xout = ml.lsim(G, U = sig, T = t)
plt.plot(T,yout)
#plt.plot(t, sig) #plotea la señal de input
plt.show()
"""
#calcular transefrencia a partir de componentes
r1=3900
r2=5600
r3=2200
r4=12000
c1=220e-9
c2=220e-9
c3=220e-9
c4=220e-9


den1=poly.Polynomial([1/(r1*r2*c1*c2), (c1+c2)/(r2*c1*c2), 1])
print (den1)

den2=poly.Polynomial([1/(r3*r4*c3*c4), (c3+c4)/(r4*c3*c4), 1])
print (den2)

r_den1 = den1.roots()
r_den2 = den2.roots()
print(r_den1)
print(r_den2)
roots = np.concatenate((r_den1,r_den2))

print(roots)

denominador = poly.polyfromroots(roots)
denominador = denominador[::-1]
print(denominador)

numerador = [1, 0, 0, 0, 0]

G = ml.tf(numerador, denominador)
print(G)
print("ceros: ")
print(ml.zero(G))
print("polos: ")
polos = ml.pole(G) 
print(polos)
ml.damp(G)
mod, fase, w = ml.bode(G)
plt.show()

"""
#chequear antitrasformada respuesta al impulso

i = 1j

beta1 = 450.53805369
gama1 = -873.84595396

beta2 = 829.67109753
gama2 = -357.65404604

z1 = gama1+beta1*i
z2 = gama2+beta2*i

A = 1/((z1-np.conj(z1))*(z1-z2)*(z1-np.conj(z2)))
C = 1/((z2-z1)*(z2-np.conj(z1))*(z2-np.conj(z2)))

mod1 = np.abs(A)
arg1 = np.angle(A)

mod2 = np.abs(C)
arg2 = np.angle(C)

G = ml.tf(numerador, denominador)
yout, T = ml.impulse(G)

plt.plot(T, yout, color = "green", label="pyhton")

gama1_2 = gama1*gama1
gama1_3 = gama1*gama1*gama1
gama1_4 = gama1*gama1*gama1*gama1
beta1_2 = beta1*beta1
beta1_3 = beta1*beta1*beta1
beta1_4 = beta1*beta1*beta1*beta1

gama2_2 = gama2*gama2
gama2_3 = gama2*gama2*gama2
gama2_4 = gama2*gama2*gama2*gama2
beta2_2 = beta2*beta2
beta2_3 = beta2*beta2*beta2
beta2_4 = beta2*beta2*beta2*beta2


inversa_laplace = 2*mod1*np.exp(gama1*T)*(np.cos(beta1*T+arg1)*(gama1_4 - 6*gama1_2 *beta1_2 + beta1_4)+ np.sin(beta1*T+arg1)*(-4*gama1_3*beta1+4*gama1*beta1_3))\
                 +2*mod2*np.exp(gama2*T)*(np.cos(beta2*T+arg2)*(gama2_4 - 6*gama2_2 *beta2_2 + beta2_4)+ np.sin(beta2*T+arg2)*(-4*gama2_3*beta2+4*gama2*beta2_3))
plt.plot(T,inversa_laplace, color = "red", label="analitic")
plt.show()

#chequear antitrasformada respuesta al escalon

i = 1j

beta1 = 450.53805369
gama1 = -873.84595396

beta2 = 829.67109753
gama2 = -357.65404604

z1 = gama1+beta1*i
z2 = gama2+beta2*i

A = 1/((z1-np.conj(z1))*(z1-z2)*(z1-np.conj(z2)))
C = 1/((z2-z1)*(z2-np.conj(z1))*(z2-np.conj(z2)))

mod1 = np.abs(A)
arg1 = np.angle(A)

mod2 = np.abs(C)
arg2 = np.angle(C)


G = ml.tf(numerador, denominador)
yout, T = ml.step(G)

plt.plot(T,yout, color = "green", label="pyhton")

gama1_2 = gama1*gama1
gama1_3 = gama1*gama1*gama1
beta1_2 = beta1*beta1
beta1_3 = beta1*beta1*beta1

gama2_2 = gama2*gama2
gama2_3 = gama2*gama2*gama2
beta2_2 = beta2*beta2
beta2_3 = beta2*beta2*beta2

inversa_laplace = 2*mod1*np.exp(gama1*T)*(np.cos(beta1*T+arg1)*(gama1_3 - 3*gama1 *beta1_2)+ np.sin(beta1*T+arg1)*(beta1_3-3*gama1_2*beta1))\
                 +2*mod2*np.exp(gama2*T)*(np.cos(beta2*T+arg2)*(gama2_3 - 3*gama2 *beta2_2)+ np.sin(beta2*T+arg2)*(beta2_3-3*gama2_2*beta2))
plt.plot(T,inversa_laplace, color = "red", label="analitic")
plt.show()

"""
#chequear antitrasformada respuesta al seno de frecuencia f

i = 1j

beta1 = 450.53805369
gama1 = -873.84595396

beta2 = 829.67109753
gama2 = -357.65404604

freq = 1000

w = 2*np.pi*freq

beta3 = w
gama3 = 0

z1 = gama1+beta1*i
z2 = gama2+beta2*i
z3 = gama3+beta3*i

A = w/((z1-np.conj(z1))*(z1-z2)*(z1-np.conj(z2))*(z1-z3)*(z1-np.conj(z3)))
B = w/((z2-z1)*(z2-np.conj(z1))*(z2-np.conj(z2))*(z2-z3)*(z2-np.conj(z3)))
C = w/((z3-z1)*(z3-np.conj(z1))*(z3-z2)*(z3-np.conj(z2))*(z3-np.conj(z3)))

mod1 = np.abs(A)
arg1 = np.angle(A)

mod2 = np.abs(B)
arg2 = np.angle(B)

mod3 = np.abs(C)
arg3 = np.angle(C)


G = ml.tf(numerador, denominador)

periodo = 1/freq #en segundos
T = np.linspace(0, 5*periodo, 1000000, endpoint=False)

sig = np.sin(w*T)
yout, T, xout = ml.lsim(G, U = sig, T = T)

plt.plot(1000*T,sig, label="señal")
plt.plot(1000*T,yout, color = "green", label="Respuesta numerica")

gama1_2 = gama1*gama1
gama1_3 = gama1*gama1*gama1
gama1_4 = gama1*gama1*gama1*gama1
beta1_2 = beta1*beta1
beta1_3 = beta1*beta1*beta1
beta1_4 = beta1*beta1*beta1*beta1

gama2_2 = gama2*gama2
gama2_3 = gama2*gama2*gama2
gama2_4 = gama2*gama2*gama2*gama2
beta2_2 = beta2*beta2
beta2_3 = beta2*beta2*beta2
beta2_4 = beta2*beta2*beta2*beta2

gama3_2 = gama3*gama3
gama3_3 = gama3*gama3*gama3
gama3_4 = gama3*gama3*gama3*gama3
beta3_2 = beta3*beta3
beta3_3 = beta3*beta3*beta3
beta3_4 = beta3*beta3*beta3*beta3

inversa_laplace = 2*mod1*np.exp(gama1*T)*(np.cos(beta1*T+arg1)*(gama1_4 - 6*gama1_2 *beta1_2 + beta1_4)+ np.sin(beta1*T+arg1)*(-4*gama1_3*beta1+4*gama1*beta1_3))\
                 +2*mod2*np.exp(gama2*T)*(np.cos(beta2*T+arg2)*(gama2_4 - 6*gama2_2 *beta2_2 + beta2_4)+ np.sin(beta2*T+arg2)*(-4*gama2_3*beta2+4*gama2*beta2_3))\
                 +2*mod3*np.exp(gama3*T)*(np.cos(beta3*T+arg3)*(gama3_4 - 6*gama3_2 *beta3_2 + beta3_4)+ np.sin(beta3*T+arg3)*(-4*gama3_3*beta3+4*gama3*beta3_3))
plt.plot(1000*T,inversa_laplace, color = "red", label="Respuesta analitica")
plt.legend()

plt.xlabel('Tiempo[ms]')
plt.ylabel('Tensión[V]')
plt.grid(linestyle='--', linewidth=1)
plt.title('Respuesta numerica vs Respuesta analitica')
plt.show()
"""
#comparar simulacion ltspice con resultados numericos
import csv

def toInt(array):
    for i in range(len(array)):
        array[i]=float(array[i])        

def toInt1000(array):
    for i in range(len(array)):
        array[i]=1000*float(array[i])        

path = "../tpAdc/csv/respuestaAlSen150"

t = []
v = []

with open(path, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter='\t')
     for row in spamreader:
         t.append(row[0])
         v.append(row[1])

t.pop(0)
v.pop(0)
toInt(v)
toInt1000(t)
plt.plot(t[:400], v[:400], label="LTSpice")


freq = 150

w = 2*np.pi*freq

periodo = 1/freq #en segundos
T = np.linspace(0, 5*periodo, 1000000, endpoint=False)

sig = np.sin(w*T)

plt.plot(1000*T,sig, label = "señal")

denominador = [1, 2381, 2.958e6, 1.987e9, 7.404e11]#transferencia de circuito propuesto

numerador = [1, 0, 0, 0, 0]

G = ml.tf(numerador, denominador)
print(G)

#yout, T = ml.step(G)
yout, T, xout = ml.lsim(G, U = sig, T = T)
plt.plot(1000*T, yout, label="Circuito propuesto")

denominador = [1, 2463, 3.033e6, 2.118e9, 7.89e11]#transferencia enunciado

numerador = [1, 0, 0, 0, 0]

G = ml.tf(numerador, denominador)
print(G)
#yout, T = ml.step(G)
yout, T, xout = ml.lsim(G, U = sig, T = T)
plt.plot(1000*T, yout, label="Enunciado")

plt.legend()

plt.xlabel('Tiempo[ms]')
plt.ylabel('Tensión[V]')
plt.grid(linestyle='--', linewidth=1)
plt.title('Comparación respuesta al escalon')
#plt.plot(1000*T,yout)
plt.show()



"""