#%% Introduccion
"""
Desarrollado por: Andres Jaramillo
Taller 3 Reconocimiento de Patrones.
Objetivos:  Visualizar e interpretar datos de acuerdo con la estadística descriptiva de los mismos.
            Diseñar e implementar clasificadores basados en inferencia Bayesiana.
"""
#%% Librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
#%% Limpiar pantalla
print('\014')
#%% Importar base de datos
Datos=np.load('D:/data.npy',allow_pickle=True)
Datos=Datos.tolist()
Datos3D=Datos["data_3D"]
Datos3D_a=Datos3D['data_a']
Datos3D_b=Datos3D['data_b']

#%% Punto 2a datos 3D
#Divida los datos aleatoriamente en conjunto de entrenamiento y prueba
#según lo considere conveniente.

Caract = np.size(Datos3D_a,axis=1)
Training_3D_A, Testing_3D_A = model_selection.train_test_split(Datos3D_a,test_size = int(0.15*len(Datos3D_a)),train_size = int(0.85*len(Datos3D_a)))
Training_3D_B, Testing_3D_B = model_selection.train_test_split(Datos3D_b,test_size = int(0.15*len(Datos3D_b)),train_size = int(0.85*len(Datos3D_b)))

#%% Punto 2c datos 3D
# Implemente un clasificador Bayesiano Gaussiano, y:

Training = np.concatenate((Training_3D_A,Training_3D_B), axis = 0) #Concatena matrices entrenamentao
Testing = np.concatenate((Testing_3D_A,Testing_3D_B), axis = 0) #Concatena matrices de prueba

Prob_A_Priori_Datos_A = len(Training_3D_A)/(len(Training_3D_A)+len(Training_3D_B))
Prob_A_Priori_Datos_B = len(Training_3D_B)/(len(Training_3D_A)+len(Training_3D_B))

u_Datos3D_a=Training_3D_A.mean(axis=0)
u_Datos3D_b=Training_3D_B.mean(axis=0)

K_Datos3D_A = np.cov(np.transpose(Training_3D_A))
K_Datos3D_B = np.cov(np.transpose(Training_3D_B))

Y_Bayessiano = np.zeros((len(Testing),1)) 
Z_Prueba = np.concatenate((np.zeros((len(Testing_3D_A),1)),np.ones((len(Testing_3D_B),1))), axis = 0)


Len=len(Testing)
for i in range(Len):
    #Prob_A_Posteriori
    Prob_A=Prob_A_Priori_Datos_A*(1/(np.sqrt(2*(np.pi**Caract)*np.linalg.det(K_Datos3D_A))))*(np.exp(-0.5*np.matmul(np.matmul(Testing[i,:]-u_Datos3D_a,np.linalg.inv(K_Datos3D_A)),Testing[i,:]-u_Datos3D_a)))
    Prob_B=Prob_A_Priori_Datos_B*(1/(np.sqrt(2*(np.pi**Caract)*np.linalg.det(K_Datos3D_B))))*(np.exp(-0.5*np.matmul(np.matmul(Testing[i,:]-u_Datos3D_b,np.linalg.inv(K_Datos3D_B)),Testing[i,:]-u_Datos3D_b)))
    Prob=(Prob_A,Prob_B)
    Indice_Img = Prob.index(max(Prob))
    if Indice_Img == 0: 
        Y_Bayessiano[i] = 0 ##Conjunto A
    elif Indice_Img == 1: 
        Y_Bayessiano[i] = 1 ##Conjunto B 
    
### Punto 2c i
# Estime la función de verosimilitud de cada clase.

DatosCorrectos_A=0
DatosCorrectos_B=0
DatosBuenos=0
for i in range(len(Y_Bayessiano)):
    if i<int(len(Y_Bayessiano)/2):
        if(Y_Bayessiano[i]==0):
            DatosCorrectos_A+=1
            DatosBuenos+=1
    else:
        if(Y_Bayessiano[i]==1):
            DatosCorrectos_B+=1
            DatosBuenos+=1

### Punto 2c iv
# Determine el error de clasificación sobre el conjunto de prueba.
Error_Bayes=1-DatosBuenos/len(Testing)
print("El error de bayes es: ",round(100*Error_Bayes,2),"%")

#%% Punto 2d datos 3D
# Implemente un clasificador Bayesiano Gaussiano Naive, y:
    
Atr = np.size(Datos3D_a,axis=1)
Bayessiano_Naive = np.zeros((len(Testing),Atr)) # salida del clasificador Bayesiano Naive    

Z_Prueba = np.concatenate((np.zeros((len(Testing_3D_A),1)),np.ones((len(Testing_3D_B),1))), axis = 0)

Std_A = Training_3D_A.std(axis = 0)
Std_B = Training_3D_B.std(axis = 0)
    
Y_Bayessiano_Naive = np.zeros((len(Testing),1))

for i in range(len(Testing)):
    # Inicialización del vector de probabilidades a posteriori: 
    P_DatosA = []
    P_DatosB = []
    for j in range(Atr):
        # Cálculo de las probabilidades a posteriori P(H|E):
        P_DatosA.append(Prob_A_Priori_Datos_A*(1/(np.sqrt(2*(np.pi**Atr))*Std_A[j]))*(np.exp(-0.5*(Testing[i,j]-u_Datos3D_a)**2/Std_A[j]**2)))                       
        P_DatosB.append(Prob_A_Priori_Datos_B*(1/(np.sqrt(2*(np.pi**Atr))*Std_B[j]))*(np.exp(-0.5*(Testing[i,j]-u_Datos3D_b)**2/Std_B[j]**2)))                       
    # Productorias: 
    P_DatosA = np.prod(P_DatosA)
    P_DatosB = np.prod(P_DatosB)
    
    # Obtención de la máxima probabilidad de pertenencia: 
    Prob = (P_DatosA,P_DatosB)
    Indice_Img = Prob.index(max(Prob))
    
    if Indice_Img == 0: 
        Y_Bayessiano_Naive[i] = 0 # Clase A
    elif Indice_Img == 1: 
        Y_Bayessiano_Naive[i] = 1 # Clase B
    
### Punto 2d i
# Estime la función de verosimilitud de cada característica para cada clase.

DatosCorrectos_A_Naive=0
DatosCorrectos_B_Naive=0
DatosBuenosNaive=0
for i in range(len(Y_Bayessiano_Naive)):
    if i<int(len(Y_Bayessiano_Naive)/2):
        if(Y_Bayessiano_Naive[i]==0):
            DatosCorrectos_A_Naive+=1
            DatosBuenosNaive+=1
    else:
        if(Y_Bayessiano_Naive[i]==1):
            DatosCorrectos_B_Naive+=1
            DatosBuenosNaive+=1

### Punto 2d iv
# Determine el error de clasificación sobre el conjunto de prueba.

Error_Bayes_Naive=1-DatosBuenosNaive/len(Testing)
print("El error de bayes naive es: ",round(100*Error_Bayes_Naive,2),"%")


#%% Taller 3
#%%Punto 1a
#Divida los datos aleatoriamente en conjunto de entrenamiento y prueba
# según lo considere conveniente.
""" --------------- Se hizo en las lineas 26 y 27 ---------------------------"""
#%%Punto 1b
#Determine m de acuerdo al método y criterio que considere pertinente.

DatosGl=np.concatenate((Training_3D_A,Training_3D_B),axis = 0) #Datos Globales Entrenamiento

#Medias de los datos (Centrarlos)
DatosCent=DatosGl-np.mean(DatosGl,axis = 0) #Datos Con la media en 0
DatosTr=np.transpose(DatosCent) # Datos transpuestos

#Matriz convarianza
K=np.cov(DatosTr)
#EigenValores y EigenVectores
EigVals, EigVects = np.linalg.eig(K)
#Porcentajes
PorcAc = 100*np.cumsum(EigVals)/sum(EigVals)
PorcInd=[]
for i in range (len(EigVals)):
    PorcInd.append(100*EigVals[i,]/sum(EigVals))
    
""" ---------------- m = 2 -------------- """
"""    Variables X y Z suman un 94,42%    """
#%%Punto 1c
#Determine la transformación de R3 a Rm utilizando PCA. (Rm = R2)
#datos salida
EigVectsPri=np.transpose(np.array([EigVects[:,0],EigVects[:,2],EigVects[:,1]])) #Se ordenan las variables de los eigen vectores 
MatrNoCorrel=EigVectsPri[:,:2] # Matriz nxm
Salida = np.transpose(np.matmul(np.transpose(MatrNoCorrel),DatosTr)) # Obtencion datos transformados
# Datos ya transformados a R2 (Conjunto entrenamiento)
NumSalida=np.size(Salida,axis=0)
# Grafica Datos transformados
#"""
plt.figure(dpi = 600)
plt.scatter(Salida[0:NumSalida//2-1,0],Salida[0:NumSalida//2-1,1],c = 'red',label = 'Datos A\'')
plt.scatter(Salida[NumSalida//2:NumSalida-1,0],Salida[NumSalida//2:NumSalida-1,1],c = 'blue',label = 'Datos B\'')

plt.title('x\' vs y\' (Datos 3D con transformacion PCA a 2D)')
plt.xlabel('x\'')
plt.ylabel('y\'') 
plt.legend()
plt.grid()   

plt.figure(dpi = 600)
plt.scatter(Salida[0:NumSalida//2-1,0],Salida[0:NumSalida//2-1,1],c = 'red',label = 'Datos A\'')

plt.title('x\' vs y\' (Datos A con transformacion PCA a 2D)')
plt.xlabel('x\'')
plt.ylabel('y\'') 
plt.legend()
plt.grid()   

plt.figure(dpi = 600)
plt.scatter(Salida[NumSalida//2:NumSalida-1,0],Salida[NumSalida//2:NumSalida-1,1],c = 'blue',label = 'Datos B\'')

plt.title('x\' vs y\' (Datos B con transformacion PCA a 2D)')
plt.xlabel('x\'')
plt.ylabel('y\'') 
plt.legend()
plt.grid()   

#"""

#%%Punto 1d
# Implemente un clasificador Bayesiano Gaussiano sobre los datos transformados, y:
#%% Punto 1d.i
#Determine la transformación de R3 a Rm utilizando PCA
# para proceder con el proceso de inferencia

Atr = np.size(Datos3D_a,axis=1)

DatosPruebaGl=np.concatenate((Testing_3D_A,Testing_3D_B),axis = 0) # Se concatenan las matrices de prueba
DatosPruebaCent=DatosPruebaGl-np.mean(DatosGl,axis = 0) # Se Centran los datos con las medias de los datos de entrenamiento 

DatosPruebaTr=np.transpose(DatosPruebaCent)
Prueba = np.transpose(np.matmul(np.transpose(MatrNoCorrel),DatosPruebaTr)) #Datos prueba transformados



# Media, K y Prob a priori datos transformados (Entrenamiento Rm)
u_DatosmD_a=Salida[0:NumSalida//2-1,:].mean(axis=0)
u_DatosmD_b=Salida[NumSalida//2:NumSalida-1,:].mean(axis=0)

K_Datos_Am = np.cov(np.transpose(Salida[0:NumSalida//2-1,:]))
K_Datos_Bm = np.cov(np.transpose(Salida[NumSalida//2:NumSalida-1,:]))

Prob_A_Priori_Datos_A_PCA = len(Salida[0:NumSalida//2-1,:])/(len(Salida[0:NumSalida//2-1,:])+len(Salida[NumSalida//2:NumSalida-1,:]))
Prob_A_Priori_Datos_B_PCA = len(Salida[NumSalida//2:NumSalida-1,:])/(len(Salida[0:NumSalida//2-1,:])+len(Salida[NumSalida//2:NumSalida-1,:]))


Y_Bayessiano_PCA = np.zeros((len(Prueba),1)) 
Z_Prueba_PCA = np.concatenate((np.zeros((len(Testing_3D_A),1)),np.ones((len(Testing_3D_B),1))), axis = 0)

Len=len(Prueba)
for i in range(Len):
    #Prob_A_Posteriori
    Prob_A=Prob_A_Priori_Datos_A_PCA*(1/(np.sqrt(2*(np.pi**Atr)*np.linalg.det(K_Datos_Am))))*(np.exp(-0.5*np.matmul(np.matmul(Prueba[i,:]-u_DatosmD_a,np.linalg.inv(K_Datos_Am)),Prueba[i,:]-u_DatosmD_a)))
    Prob_B=Prob_A_Priori_Datos_B_PCA*(1/(np.sqrt(2*(np.pi**Atr)*np.linalg.det(K_Datos_Bm))))*(np.exp(-0.5*np.matmul(np.matmul(Prueba[i,:]-u_DatosmD_b,np.linalg.inv(K_Datos_Bm)),Prueba[i,:]-u_DatosmD_b)))
    Prob_PCA=(Prob_A,Prob_B)
    Indice_Img = Prob_PCA.index(max(Prob_PCA))
    if Indice_Img == 0: 
        Y_Bayessiano_PCA[i] = 0 ##Conjunto A
    elif Indice_Img == 1: 
        Y_Bayessiano_PCA[i] = 1 ##Conjunto B 

#%% Punto 1d.ii
# Estime la función de verosimilitud de cada clase.

DatosCorrectos_A=0
DatosCorrectos_B=0
for i in range(len(Y_Bayessiano_PCA)):
    if i<int(len(Y_Bayessiano_PCA)/2):
        if(Y_Bayessiano_PCA[i]==0):
            DatosCorrectos_A+=1
    else:
        if(Y_Bayessiano_PCA[i]==1):
            DatosCorrectos_B+=1

Verosimilitud_A_PCA=DatosCorrectos_A*Prob_A_Priori_Datos_A_PCA/(DatosCorrectos_A*Prob_A_Priori_Datos_A_PCA+DatosCorrectos_B*Prob_A_Priori_Datos_B_PCA)
Verosimilitud_B_PCA=DatosCorrectos_B*Prob_A_Priori_Datos_B_PCA/(DatosCorrectos_A*Prob_A_Priori_Datos_A_PCA+DatosCorrectos_B*Prob_A_Priori_Datos_B_PCA)

#%% Punto 1d.iii
# Grafique la clasificación realizada.

Datos_AX_Pri=[]
Datos_AY_Pri=[]

Datos_BX_Pri=[]
Datos_BY_Pri=[]


for i in range(len(Y_Bayessiano_PCA)):
    if Y_Bayessiano_PCA[i] == 0: 
        Datos_AX_Pri.append(Prueba[i,0]) #Grupo A Eje X primo
        Datos_AY_Pri.append(Prueba[i,1]) #Grupo A Eje Z primo
    elif Y_Bayessiano_PCA[i] == 1: 
        Datos_BX_Pri.append(Prueba[i,0]) #Grupo B Eje X primo
        Datos_BY_Pri.append(Prueba[i,1]) #Grupo B Eje Z primo

plt.figure(dpi = 600)
plt.scatter(Datos_AX_Pri,Datos_AY_Pri,c='red',label = 'Datos 3D -> 2D conjunto A\' (Datos de Prueba)')
plt.scatter(Datos_BX_Pri,Datos_BY_Pri,c='blue',label = 'Datos 3D -> 2D conjunto \'B (Datos de Prueba)')

plt.title('x\' vs y\' (Datos Prueba 3D con transformacion PCA a 2D)')
plt.xlabel('x\'')
plt.ylabel('y\'') 
plt.legend()
plt.grid()   

plt.figure(dpi = 600)
plt.scatter(Datos_AX_Pri,Datos_AY_Pri,c='red',label = 'Datos 3D -> 2D conjunto A\' (Datos de Prueba)')

plt.title('x\' vs y\' (Datos Prueba 3D con transformacion PCA a 2D)')
plt.xlabel('x\'')
plt.ylabel('y\'') 
plt.legend()
plt.grid()   

plt.figure(dpi = 600)
plt.scatter(Datos_BX_Pri,Datos_BY_Pri,c='blue',label = 'Datos 3D -> 2D conjunto \'B (Datos de Prueba)')

plt.title('x\' vs y\' (Datos Prueba 3D con transformacion PCA a 2D)')
plt.xlabel('x\'')
plt.ylabel('y\'') 
plt.legend()
plt.grid()   

#%% Punto 1d.iv
# Determine el error de clasificación.

Err_Bayes_PCA = 100*sum(Y_Bayessiano_PCA != Z_Prueba_PCA)/len(Y_Bayessiano_PCA)
print("El error de bayes con PCA es: ",round(Err_Bayes_PCA[0],2),'%')
