import numpy as np
import time
import math
import matplotlib.pyplot as plt
import copy

# Funciones de lectura y preparación de datos
def Lectura_De_Archivo():
    Nombre_Archivo = ("instancias creadas/CMT1-5VTH1.txt")
    archivo = open(Nombre_Archivo, "r")
    N = int(archivo.readline())
    v = int(archivo.readline())
    Capacidad_Vehiculos = int(archivo.readline())
    Tiempo_Maximo_Vehiculo = int(archivo.readline())
    Requerimientos = np.loadtxt(Nombre_Archivo, skiprows=4 + N).astype(int)
    Matriz_Coordenadas = np.loadtxt(Nombre_Archivo, skiprows=4, max_rows=N)
    Matriz = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            x1 = Matriz_Coordenadas[i][1]
            y1 = Matriz_Coordenadas[i][2]
            x2 = Matriz_Coordenadas[j][1]
            y2 = Matriz_Coordenadas[j][2]
            distancia = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            velocidad = 50  # km/h
            tiempo_minutos = round((distancia / velocidad * 60) + 0.5)
            Matriz[i][j] = tiempo_minutos
    archivo.close()
    return (N, v, Capacidad_Vehiculos, Tiempo_Maximo_Vehiculo, Matriz, Requerimientos, Matriz_Coordenadas)

# Funciones de optimización y búsqueda
def calcular_ahorros(Matriz_Coordenadas):
    n = len(Matriz_Coordenadas)
    ahorros = []
    for i in range(1, n):
        for j in range(i + 1, n):
            ahorro = Matriz_Coordenadas[i][0] + Matriz_Coordenadas[0][j] - Matriz_Coordenadas[i][j]
            ahorros.append((i, j, ahorro))
    return sorted(ahorros, key=lambda x: -x[2])

def metodo_de_ahorros(N, Tiempos, Requerimientos, Capacidad_Vehiculos):
    ahorros = calcular_ahorros(Tiempos)
    rutas = [[i] for i in range(1, N)]
    capacidades = [Requerimientos[i] for i in range(1, N)]

    for i, j, _ in ahorros:
        ruta_i = next(ruta for ruta in rutas if i in ruta)
        ruta_j = next(ruta for ruta in rutas if j in ruta)

        if ruta_i != ruta_j:
            capacidad_total = sum(Requerimientos[nodo] for nodo in ruta_i + ruta_j)
            if capacidad_total <= Capacidad_Vehiculos:
                nueva_ruta = ruta_i + ruta_j
                rutas.remove(ruta_i)
                rutas.remove(ruta_j)
                rutas.append(nueva_ruta)

    rutas = [[0] + ruta + [0] for ruta in rutas]
    return rutas

# Funciones de cálculo de costos
def calcular_costo(solucion, Tiempos):
    costo = 0
    for i in range(len(solucion) - 1):
        costo += Tiempos[solucion[i]][solucion[i + 1]]
    return costo

def calcular_costo_total(rutas, Tiempos):
    return sum(calcular_costo(ruta, Tiempos) for ruta in rutas)

# Funciones de búsqueda de vecindarios y búsquedas Tabú

def vecindario_swap_2(solucion, Tiempos):
    vecindario = []
    n = len(solucion)

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            if i == 1 and j == n - 1:
                continue

            vecina = copy.deepcopy(solucion)
            vecina[i], vecina[j] = vecina[j], vecina[i]
            vecindario.append((vecina, (i, j)))

    return vecindario

def busqueda_tabu(Tiempos, iteraciones_maximas, tamano_lista_tabu, solucion_inicial):
    mejor_solucion = solucion_inicial
    mejor_costo = calcular_costo(mejor_solucion, Tiempos)
    lista_tabu = []

    for iteracion in range(iteraciones_maximas):
        vecindario = vecindario_swap_2(mejor_solucion, Tiempos)
        mejor_vecina = None
        mejor_vecina_costo = float('inf')
        mejor_intercambio = None

        for vecina, intercambio in vecindario:
            if vecina[0] != mejor_solucion[0] or vecina[-1] != mejor_solucion[-1]:
                continue

            if intercambio in lista_tabu:
                continue

            costo_vecina = calcular_costo(vecina, Tiempos)
            if costo_vecina < mejor_vecina_costo:
                mejor_vecina = vecina
                mejor_vecina_costo = costo_vecina
                mejor_intercambio = intercambio

        if mejor_vecina is None:
            break

        mejor_solucion = mejor_vecina
        mejor_costo = mejor_vecina_costo
        lista_tabu.append(mejor_intercambio)

        if len(lista_tabu) > tamano_lista_tabu:
            lista_tabu.pop(0)

    return mejor_solucion, mejor_costo

def two_opt(route, distance_matrix):
    def calculate_distance(route, distance_matrix):
        distance = 0
        for i in range(len(route) - 1):
            distance += distance_matrix[route[i]][route[i + 1]]
        return distance

    improved = True
    while improved:
        improved = False
        best_distance = calculate_distance(route, distance_matrix)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_distance = calculate_distance(new_route, distance_matrix)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
    return route, best_distance
def asignar_clusters_a_vehiculos(clusters, Tiempos, v, Tiempo_Maximo):
    vehiculos_asignados = [[] for _ in range(v)]
    tiempos_vehiculos = [0] * v

    for cluster in clusters:
        tiempo_asignado = calcular_costo(cluster, Tiempos)

        vehiculo_disponible = None
        for vehiculo in range(v):
            if tiempos_vehiculos[vehiculo] + tiempo_asignado <= Tiempo_Maximo:
                vehiculo_disponible = vehiculo
                break

        if vehiculo_disponible is not None:
            vehiculos_asignados[vehiculo_disponible].append(cluster)
            tiempos_vehiculos[vehiculo_disponible] += tiempo_asignado
        else:
            print("No es posible asignar todos los clusters a los vehículos.")
            return None

    return vehiculos_asignados

# Funciones de visualización
def graficar_solucion(rutas, coordenadas, title):
    plt.figure(figsize=(10, 8))

    for ruta in rutas:
        x = [coordenadas[i][1] for i in ruta]
        y = [coordenadas[i][2] for i in ruta]
        plt.plot(x, y, marker='o', linestyle='-')

    plt.title(title)
    plt.grid(True)
    plt.show()

# Función mejorada de swap inter rutas
def swap_inter_rutas(rutas, Tiempos):
    def calcular_costo_ruta(ruta, Tiempos):
        costo = 0
        for k in range(len(ruta) - 1):
            costo += Tiempos[ruta[k]][ruta[k + 1]]
        return costo

    def calcular_costo_total(rutas, Tiempos):
        return sum(calcular_costo_ruta(ruta, Tiempos) for ruta in rutas)

    n_rutas = len(rutas)
    mejor_costo_total = calcular_costo_total(rutas, Tiempos)
    mejor_rutas = copy.deepcopy(rutas)

    for i in range(n_rutas):
        for j in range(i + 1, n_rutas):
            for a in range(1, len(rutas[i]) - 1):
                for b in range(1, len(rutas[j]) - 1):
                    nuevas_rutas = copy.deepcopy(rutas)
                    nuevas_rutas[i][a], nuevas_rutas[j][b] = nuevas_rutas[j][b], nuevas_rutas[i][a]
                    graficar_solucion(nuevas_rutas, Coordenadas, "swap")
                    costo_total_nuevas = calcular_costo_total(nuevas_rutas, Tiempos)

                    if costo_total_nuevas < mejor_costo_total:
                        mejor_costo_total = costo_total_nuevas
                        mejor_rutas = nuevas_rutas

    return mejor_rutas, mejor_costo_total

def asignar_clusters_a_vehiculos(clusters, Tiempos, num_vehiculos, Tiempo_Maximo_Vehiculo):
    vehiculos = [[] for _ in range(num_vehiculos)]
    tiempos = [0] * num_vehiculos
    for cluster in clusters:
        cluster_tiempo = calcular_costo(cluster, Tiempos)
        min_tiempo_idx = tiempos.index(min(tiempos))
        if tiempos[min_tiempo_idx] + cluster_tiempo <= Tiempo_Maximo_Vehiculo:
            vehiculos[min_tiempo_idx].append(cluster)
            tiempos[min_tiempo_idx] += cluster_tiempo
        else:
            return []
    return vehiculos

# Código de prueba para usar la función actualizada
N, v, Capacidad_Vehiculos, Tiempo_Maximo_Vehiculo, Tiempos, Requerimientos, Coordenadas = Lectura_De_Archivo()
max_intentos_sin_mejora = 10

rutas = metodo_de_ahorros(N, Tiempos, Requerimientos, Capacidad_Vehiculos)
print(rutas)
graficar_solucion(rutas, Coordenadas, "Grafico Inicial")

Mejor_Costo = float('inf')
tiempo_inicio = time.time()

rutas_TS = []
Costo_TS = 0
for ruta in rutas:
    ruta_TS, Costo_cluster_TS = busqueda_tabu(Tiempos, 1000, 5, ruta)
    Costo_TS += Costo_cluster_TS
    rutas_TS.append(ruta_TS)

rutas_2OPT = []
Costo_2OPT = 0
for ruta in rutas_TS:
    ruta_2OPT, Costo_cluster_2OPT = two_opt(ruta, Tiempos)
    Costo_2OPT += Costo_cluster_2OPT
    rutas_2OPT.append(ruta_2OPT)
Mejor_Costo = Costo_2OPT
for i in range(100):
    rutas_inter_swap, Costo_inter_swap = swap_inter_rutas(rutas_2OPT, Tiempos)
    if Costo_inter_swap < Mejor_Costo:
        Mejor_Costo = Costo_inter_swap
        Mejor_Ruta = rutas_inter_swap
    graficar_solucion(rutas_inter_swap,Coordenadas,"solucion")

Asignacion_Vehiculos = asignar_clusters_a_vehiculos(Mejor_Ruta, Tiempos, v, Tiempo_Maximo_Vehiculo)
tiempo_total = time.time() - tiempo_inicio

#graficar_solucion_2OPT(Mejor_Ruta, Coordenadas)
print("Mejor Costo: ", Mejor_Costo)
print("Tiempo de ejecución: ", tiempo_total, "segundos")