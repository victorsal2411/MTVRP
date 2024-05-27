import math
import numpy as np
Nombre_Archivo="CMT11.txt"
archivo=open("instancias creadas/"+Nombre_Archivo,"r");
instancia=open("instancias creadas/Tiempos_"+Nombre_Archivo,"w")
N = int(archivo.readline())
v = int(archivo.readline())

Nodos = list()
Matriz_Coordenadas=np.loadtxt("instancias creadas/"+Nombre_Archivo,skiprows=4,max_rows=N)
Matriz=np.zeros((N,N))

instancia.write("param n:="+str(N))
instancia.write(";\n")
instancia.write("param v:="+str(v)+";\n\n")

instancia.write("param T:\n")


for i in range (N):
    instancia.write ("\t")
    instancia.write(str(i))
instancia.write("=\n")

for i in range (N):
    instancia.write(str(i))
    instancia.write("\t")
    for j in range (N):
        x1 = Matriz_Coordenadas[i][1]
        y1 = Matriz_Coordenadas[i][2]
        x2 = Matriz_Coordenadas[j][1]
        y2 = Matriz_Coordenadas[j][2]
        distancia = int(math.sqrt((x2-x1)**2+(y2-y1)**2))
        velocidad = 50  # km/h
        tiempo_minutos = round((distancia / velocidad * 60)+0.5)

        if distancia!=0:
            instancia.write(str(tiempo_minutos))
        else: 
            instancia.write(".")
        instancia.write("\t")
    instancia.write("\n")

instancia.write(";")
instancia.write("\n\n")
Requerimientos=np.loadtxt("instancias creadas/"+Nombre_Archivo,skiprows=N+4)

instancia.write("param Q:=\n")
for i in range (N):
    instancia.write(str(i))
    instancia.write("\t")
    instancia.write(str(Requerimientos[i]))
    instancia.write("\n")


print("______________________________:")
#paso 3 hacemos el calculo segun la formula dada.