param r;
param n;
param v;

set V := {1 .. v};
set N := {0 .. n};
set R := {0 .. n-1};
set A := {i in N, j in N : i<>j}; #A

param T{A};		#tiempo en minutos dentro de la matriz
param Tiempo_maximo = 1146;
param Q{N diff {0}};
param Capacidad = 200;

var X{A,V,R} binary;	# 1 si vv V pasa por el arco E en la rr R
var Y{V,R,N} binary;	# 1 si vv V visita el nodo N en la rr R
var q{A,V,R} >= 0;

minimize Z: sum{(i,j) in A} T[i,j] * (sum{vv in V, rr in R} X[i,j,vv,rr]);
s.t.
R1{i in N diff{0}}: 
	sum{vv in V, rr in R} Y[vv,rr,i] = 1; # 1 nodo es visitado por 1 solo vv en una rr
R2{i in N, vv in V, rr in R}: 
	sum{j in N: i<>j} X[i,j,vv,rr] = Y[vv,rr,i]; # Nodo i tiene una unica entrada
R3{i in N, vv in V, rr in R}: 
	sum{j in N: i<>j} X[j,i,vv,rr] = Y[vv,rr,i]; # Nodo i tiene una unica salida
R4{i in N diff{0}}:
	sum{(i,j) in A, vv in V, rr in R} q[j,i,vv,rr] - sum{(i,j) in A, vv in V, rr in R} q[i,j,vv,rr] = Q[i];
R5{(i,j) in A, vv in V, rr in R}: 
	q[i,j,vv,rr] <= Capacidad*X[i,j,vv,rr];
R7{vv in V}:
	sum{rr in R, (i,j) in A} T[i,j]*X[i,j,vv,rr] <= Tiempo_maximo;	# Tiempo de la sumatoria de los viajes de un vehículo no supera el limite de tiempo
	