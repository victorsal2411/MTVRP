import numpy as np
import time
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from itertools import permutations
import copy

inicio = time.time()

tiempo_inicio = time.time()


def Lectura_De_Archivo():
    #Nombre_Archivo = ("instancias creadas/CMT3-1V.txt")
    #Nombre_Archivo = ("instancias creadas/CMT2-1V.txt")
    #Nombre_Archivo = ("instancias creadas/CMT11.txt")
    Nombre_Archivo = ("instancias creadas/CMT1-5VTH1.txt")
    # Guardado cantidad de codos, cantidad de vehículos, capacidad máxima de los vehículos y tiempo máximo de los vehículos
    archivo = open(Nombre_Archivo, "r")
    N = int(archivo.readline())
    v = int(archivo.readline())
    Capacidad_Vehiculos = int(archivo.readline())
    Tiempo_Maximo_Vehiculo = int(archivo.readline())
    Requerimientos = np.loadtxt(Nombre_Archivo, skiprows=4 + N)
    Requerimientos = Requerimientos.astype(int)
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
    # Matriz_Coordenadas = Matriz_Coordenadas[:, 1:]

    return (N, v, Capacidad_Vehiculos, Tiempo_Maximo_Vehiculo, Matriz, Requerimientos, Matriz_Coordenadas)
def Clustering(coordenadas, k):

    indices = coordenadas[:, 0].astype(int)  # Obtener los índices del nodo como enteros
    deposito = coordenadas[0, 1:]
    clientes = coordenadas[1:, 1:]

    kmeans = KMeans(n_clusters=k, n_init=10)
    #ejecutar 10 veces cada instancia con parametro 100%aleatorio y sacar minimo, maximo y promedio
    kmeans.fit(clientes)
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_

    indices_clusters = []  # Arreglo para guardar los índices de cada cluster
    for i in range(k):
        cluster_indices = indices[1:][etiquetas == i]  # Obtener los índices correspondientes al cluster
        cluster_indices_with_depot = np.concatenate(([0], cluster_indices)).astype(
            int)  # Agregar el nodo 0 al inicio y convertir a enteros
        indices_clusters.append(cluster_indices_with_depot.tolist())  # Guardar los índices del cluster en el arreglo

    # Agregar un 0 al final de cada lista en indices_clusters
    indices_clusters = [cluster_indices + [0] for cluster_indices in indices_clusters]
    """
    print("Indices clusters: ", indices_clusters)

    # Generar el gráfico de dispersión
    colores = ['red', 'green', 'blue', 'yellow', 'orange']
    # Graficar el depósito
    plt.scatter(deposito[0], deposito[1], c='black', marker='s', label='Depot')
    # Graficar los clusters
    for i in range(k):
        cluster_points = clientes[etiquetas == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[i], label='Cluster {}'.format(i + 1))
    # Graficar los centroides
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='x', color='black', label='Centroid')

    #plt.xlabel('Eje X')
    #plt.ylabel('Eje Y')
    plt.title('Initial Clusters')
    plt.legend()
    plt.show()
    """
    return indices_clusters, centroides
def Sub_Clustering(N,coordenadas, k, nodos_cluster):
    coordenadas_cluster = coordenadas.copy()  # Haz una copia para no modificar el array original

    # Mantener solo las filas donde el primer elemento está en nodos_cluster
    coordenadas_cluster = np.array([fila for fila in coordenadas_cluster if fila[0] in nodos_cluster])

    coordenadas = coordenadas_cluster

    # Asegúrate de ajustar el código según tus necesidades específicas
    indices = coordenadas[:, 0].astype(int)
    deposito = coordenadas[0, 1:]
    clientes = coordenadas[1:, 1:]
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(clientes)
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_

    indices_clusters = []  # Arreglo para guardar los índices de cada cluster
    for i in range(k):
        cluster_indices = indices[1:][etiquetas == i]  # Obtener los índices correspondientes al cluster
        cluster_indices_with_depot = np.concatenate(([0], cluster_indices)).astype(
            int)  # Agregar el nodo 0 al inicio y convertir a enteros
        indices_clusters.append(cluster_indices_with_depot.tolist())  # Guardar los índices del cluster en el arreglo

    # Agregar un 0 al final de cada lista en indices_clusters
    indices_clusters = [cluster_indices + [0] for cluster_indices in indices_clusters]
    """
    # Generar el gráfico de dispersión
    colores = ['red', 'green', 'blue', 'yellow', 'orange']
    # Graficar el depósito
    plt.scatter(deposito[0], deposito[1], c='black', marker='s', label='Depósito')
    # Graficar los clusters
    for i in range(k):
        cluster_points = clientes[etiquetas == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[i], label='Cluster {}'.format(i + 1))
    # Graficar los centroides
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='x', color='black', label='Centroides')

    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Gráfico de Clusters')
    plt.legend()
    plt.show()"""

    return indices_clusters
def verificar_capacidad(cluster_indices, Requerimientos, Capacidad_Vehiculos):
    # Calcular la sumatoria de los requerimientos para los índices en cluster_indices
    num_clusters=len(cluster_indices)
    clusters_factibles = [True] * num_clusters
    for i in range(num_clusters):
        suma_requerimientos = sum(Requerimientos[nodo] for nodo in cluster_indices[i])
        if suma_requerimientos > Capacidad_Vehiculos:
            clusters_factibles[i] = False
    return clusters_factibles
def encontrar_ruta_mas_corta(indices, Tiempos):
    # Inicializar la ruta con el nodo 0
    ruta = [0]
    indices_ruta = [0]
    # Obtener el índice del nodo actual
    nodo_actual = 0
    # Iterar hasta que se visiten todos los nodos
    while len(ruta) < len(indices):
        # Inicializar la distancia mínima y el nodo más cercano
        distancia_minima = float('inf')
        nodo_cercano = None
        # Buscar el nodo más cercano no visitado
        for i in range(len(indices)):
            if indices[i] not in indices_ruta:
                distancia = Tiempos[nodo_actual][i]
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    nodo_cercano = indices[i]
        # Agregar el nodo más cercano a la ruta
        if nodo_cercano is not None:
            indices_ruta.append(nodo_cercano)
            ruta.append(nodo_cercano)
            nodo_actual = nodo_cercano
        else:
            break

    # Agregar el nodo 0 al final de la ruta
    indices_ruta.append(0)
    ruta.append(0)

    # Calcular la distancia total de la ruta
    costo_total = 0
    for i in range(len(ruta) - 1):
        costo_total = costo_total + (Tiempos[ruta[i]][ruta[i + 1]])

    return indices_ruta, costo_total
def calcular_costo(solucion, Tiempos):
    costo = 0
    for i in range(len(solucion) - 1):
        costo += Tiempos[solucion[i]][solucion[i + 1]]
    # Agregar el costo de regreso al depósito (nodo 0)
    costo += Tiempos[solucion[-1]][solucion[0]]
    return costo
def vecindario_swap(solucion):
    vecindario = []
    n = len(solucion)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            vecina = copy.deepcopy(solucion)
            vecina[i], vecina[j] = vecina[j], vecina[i]
            vecindario.append(vecina)
    return vecindario
def vecindario_swap_2(solucion, Tiempos):
    vecindario = []
    n = len(solucion)

    # Calcular el costo promedio de los arcos en el vecindario
    costo_promedio_vecindario = sum(Tiempos[solucion[i - 1]][solucion[i]] for i in range(1, n)) / (n - 1)

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            if i == 1 and j == n - 1:
                # Evitar arcos (0,0) asegurando que cada solución inicia y termina en 0
                continue

            vecina = copy.deepcopy(solucion)
            vecina[i], vecina[j] = vecina[j], vecina[i]

            # Calcular el costo promedio de los arcos en la solución vecina
            costo_promedio_vecina = sum(Tiempos[vecina[i - 1]][vecina[i]] for i in range(1, n)) / (n - 1)

            # Si el costo promedio de la vecina es menor o igual al costo promedio del vecindario, agregarla
            if costo_promedio_vecina <= costo_promedio_vecindario:
                vecindario.append(vecina)

    return vecindario
def busqueda_tabu(Tiempos, iteraciones_maximas, tamano_lista_tabu, solucion_inicial):
    mejor_solucion = solucion_inicial
    mejor_costo = calcular_costo(mejor_solucion, Tiempos)
    lista_tabu = []

    for iteracion in range(iteraciones_maximas):
        vecindario = vecindario_swap_2(mejor_solucion, Tiempos)
        mejor_vecina = None
        mejor_vecina_costo = float('inf')

        for vecina in vecindario:
            # Asegurar que el primer y último elemento no cambien
            if vecina[0] != mejor_solucion[0] or vecina[-1] != mejor_solucion[-1]:
                continue

            costo_vecina = calcular_costo(vecina, Tiempos)
            if costo_vecina < mejor_vecina_costo and vecina not in lista_tabu:
                mejor_vecina = vecina
                mejor_vecina_costo = costo_vecina

        if mejor_vecina is None:
            break

        mejor_solucion = mejor_vecina
        mejor_costo = mejor_vecina_costo
        lista_tabu.append(mejor_vecina)

        if len(lista_tabu) > tamano_lista_tabu:
            lista_tabu.pop(0)

    return mejor_solucion, mejor_costo
def graficar_solucion_KNN(cluster_indices, coordenadas):
    plt.figure(figsize=(10, 8))

    for cluster in cluster_indices:
        x = [coordenadas[i][1] for i in cluster]
        y = [coordenadas[i][2] for i in cluster]

        plt.scatter(x, y, marker='o')

    plt.title('Initial solution KNN')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')

    plt.grid(True)
    plt.show()
def graficar_solucion_TS(cluster_indices, coordenadas):
    plt.figure(figsize=(10, 8))

    for cluster in cluster_indices:
        x = [coordenadas[i][1] for i in cluster]
        y = [coordenadas[i][2] for i in cluster]

        # Agregar el punto de inicio al final para cerrar el ciclo
        x.append(x[0])
        y.append(y[0])

        plt.plot(x, y, marker='o', linestyle='-')

    plt.title('Tabu search solution')
    #plt.xlabel('Coordenada X')
    #plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.show()
def graficar_solucion_2OPT(cluster_indices, coordenadas):
    plt.figure(figsize=(10, 8))

    for cluster in cluster_indices:
        x = [coordenadas[i][1] for i in cluster]
        y = [coordenadas[i][2] for i in cluster]

        # Agregar el punto de inicio al final para cerrar el ciclo
        x.append(x[0])
        y.append(y[0])

        plt.plot(x, y, marker='o', linestyle='-')

    plt.title('2OPT Solution')
    #plt.xlabel('Coordenada X')
    #plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.show()
def calculate_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    return distance
def two_opt(route, distance_matrix):
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

        # Si se encontró un vehículo disponible, asignar el cluster
        if vehiculo_disponible is not None:
            vehiculos_asignados[vehiculo_disponible].append(cluster)
            tiempos_vehiculos[vehiculo_disponible] += tiempo_asignado
        else:
            # Si no se encontró un vehículo disponible, indicar que no es posible asignar todos los clusters
            #print("No es posible asignar todos los clusters a los vehículos.")
            return None
    # Revisar y ajustar asignaciones si es necesario
    #for i in range (v):
        #print (tiempos_vehiculos[i])
    return vehiculos_asignados


N, v, Capacidad_Vehiculos, Tiempo_Maximo_Vehiculo, Tiempos, Requerimientos, Coordenadas = Lectura_De_Archivo()
max_intentos_sin_mejora =1000
N_Clusters =3

Mejor_Costo=float('inf')

for intento in range(max_intentos_sin_mejora):

    cluster_indices, Centroides = Clustering(Coordenadas, N_Clusters)

    verificacion = verificar_capacidad(cluster_indices, Requerimientos, Capacidad_Vehiculos)
    #print(verificacion)
    while not all(verificacion):
        for i in range(len(verificacion)):
            if not verificacion[i]:
                nuevo_cluster = Sub_Clustering(N, Coordenadas, 2, cluster_indices[i])
                del cluster_indices[i]
                cluster_indices.extend(nuevo_cluster)
                verificacion = verificar_capacidad(cluster_indices, Requerimientos, Capacidad_Vehiculos)

    #print(cluster_indices)
    N_Clusters=len(cluster_indices)

    rutas_KNN = []
    Costo_KNN = 0
    for i in range(N_Clusters):
        ruta_mas_corta_KNN, Costo_cluster_i_KNN = encontrar_ruta_mas_corta(cluster_indices[i], Tiempos)
        Costo_KNN += Costo_cluster_i_KNN
        rutas_KNN.append(ruta_mas_corta_KNN)

    #print("KNN: ", rutas_KNN)
    #print("Distancia KNN: ", Costo_KNN)
    #graficar_solucion_KNN(rutas_KNN, Coordenadas)

    rutas_TS = []
    Costo_TS = 0
    for i in range(N_Clusters):
        ruta_TS, Costo_cluster_TS = busqueda_tabu(Tiempos, 1000, 5, rutas_KNN[i])
        Costo_TS += Costo_cluster_TS
        rutas_TS.append(ruta_TS)
    #print("TS: ", rutas_TS)
    #print("Distancia TS: ", Costo_TS)
    #graficar_solucion_TS(rutas_TS, Coordenadas)
    tiempo_fin = time.time()

    rutas_2OPT = []
    Costo_2OPT = 0
    for i in range(N_Clusters):
        ruta_2OPT, Costo_cluster_2OPT = two_opt(rutas_TS[i], Tiempos)
        Costo_2OPT += Costo_cluster_2OPT
        rutas_2OPT.append(ruta_2OPT)
    #print("2OPT: ", rutas_2OPT)
    #print("Distancia 2OPT: ", Costo_2OPT)
    graficar_solucion_2OPT(rutas_2OPT, Coordenadas)

    Asignacion_Vehiculos=asignar_clusters_a_vehiculos(rutas_2OPT, Tiempos, v,Tiempo_Maximo_Vehiculo)
    #print(Asignacion_Vehiculos)
    tiempo_fin = time.time()

    if (Costo_2OPT<= Mejor_Costo):
        Mejor_knn = Costo_KNN
        Mejor_TS = Costo_TS
        Mejor_Costo = Costo_2OPT
        Mejor_Ruta = rutas_2OPT
        Costo_por_vehiculo=Asignacion_Vehiculos
    #print(Costo_2OPT)

tiempo_total = tiempo_fin - tiempo_inicio
graficar_solucion_2OPT(Mejor_Ruta, Coordenadas)
graficar_solucion_KNN(rutas_KNN, Coordenadas)
print("Mejor knn: ", Mejor_knn)
print("Mejor TS: ", Mejor_TS)
print("Mejor Costo: ",Mejor_Costo)
print(Costo_por_vehiculo)
print("Tiempo de ejecución:", tiempo_total, "segundos")