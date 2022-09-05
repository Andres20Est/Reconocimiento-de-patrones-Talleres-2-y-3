#%% Introduccion
"""
Desarrollado por: Andres Jaramillo
Taller 1 Reconocimiento de Patrones.
Objetivos:  Visualizar e interpretar datos de acuerdo con la estadística descriptiva de los mismos.
            Diseñar e implementar clasificadores basados en inferencia Bayesiana.
"""
#%% Librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
#%% Limpiar pantalla
print('\014')
#%% Importar base de datos
Datos=np.load('D:/data.npy',allow_pickle=True)
Datos=Datos.tolist()
Datos2D=Datos["data_2D"]
Datos2D_a=Datos2D['data_a']
Datos2D_b=Datos2D['data_b']
Datos2D_a=np.transpose(Datos2D_a)
Datos2D_b=np.transpose(Datos2D_b)

Datos3D=Datos["data_3D"]
Datos3D_a=Datos3D['data_a']
Datos3D_b=Datos3D['data_b']

#%% Punto 1a  datos 2D
# Grafique los datos utilizando un color distintivo para cada clase.
#"""
plt.figure(dpi=600)
plt.scatter(Datos2D_a[:,0],Datos2D_a[:,1],c='red',label = 'Datos 2D A')
plt.scatter(Datos2D_b[:,0],Datos2D_b[:,1],c='blue',label = 'Datos 2D B')
plt.title('Comparativa Datos 2D')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid()

plt.figure(dpi=600)
plt.scatter(Datos2D_a[:,0],Datos2D_a[:,1],c='red',label = 'Datos 2D A')
plt.title('Grafica unicamente Datos 2D A')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid()

plt.figure(dpi=600)
plt.scatter(Datos2D_b[:,0],Datos2D_b[:,1],c='blue',label = 'Datos 2D B')
plt.title('Grafica unicamente Datos 2D B')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid()
#"""

#%% Punto 1b  datos 2D
# Determine el centro de cada clase ua y ub
Media_2D_A=np.mean(Datos2D_a,axis=0)
Media_2D_B=np.mean(Datos2D_b,axis=0)
print('La media de los datos a es: ',Media_2D_A,'\n','La media de los datos b es: ',Media_2D_B)
print('\n')

#%% Punto 1c  datos 2D
# Determine las matrices de covarianza de cada clase {Ka,Kb}. ¿Que se puede concluir?
K_Datos2D_A=np.cov(np.transpose(Datos2D_a))
K_Datos2D_B=np.cov(np.transpose(Datos2D_b))

print('La matriz de covarianza de los datos a es: ',K_Datos2D_A)
print('\n')
print('La matriz de covarianza de los datos b es: ',K_Datos2D_B)
print('\n')

#%% Punto 1d  datos 2D
#Determine y grafique el histograma de los datos.
#¿Los datos siguen alguna distribución de probabilidad en particular? (Gaussiana)
#"""
plt.figure(dpi=600)
plt.hist2d(Datos2D_a[:,0],Datos2D_a[:,1], bins = (50,50))
plt.colorbar()
plt.title('Histograma Datos 2D A')
plt.xlabel('Datos 2D A, Eje X')
plt.ylabel('Datos 2D A, Eje Y')
plt.legend()
plt.grid()

plt.figure(dpi=600)
plt.title('Histograma datos 2D_A Eje X ')
plt.hist(Datos2D_a[:,0], bins = 60)
plt.grid()
plt.show()
plt.clf()

plt.figure(dpi=600)
plt.title('Histograma datos 2D_A Eje Y ')
plt.hist(Datos2D_a[:,1], bins = 60)
plt.grid()
plt.show()
plt.clf()

plt.figure(dpi=600)
plt.hist2d(Datos2D_b[:,0],Datos2D_b[:,1], bins = (50,50))
plt.colorbar()
plt.title('Histograma Datos 2D B')
plt.xlabel('Datos 2D B, Eje X')
plt.ylabel('Datos 2D B, Eje Y')
plt.legend()
plt.grid()

plt.figure(dpi=600)
plt.title('Histograma datos 2D_B Eje X ')
plt.hist(Datos2D_b[:,0], bins = 60)
plt.grid()
plt.show()
plt.clf()

plt.figure(dpi=600)
plt.title('Histograma datos 2D_B Eje Y ')
plt.hist(Datos2D_b[:,1], bins = 60)
plt.grid()
plt.show()
plt.clf()
#"""

#%% Punto 2a datos 3D
#Divida los datos aleatoriamente en conjunto de entrenamiento y prueba
#según lo considere conveniente.

Caract = np.size(Datos3D_a,axis=1)
Training_3D_A, Testing_3D_A = model_selection.train_test_split(Datos3D_a,test_size = int(0.15*len(Datos3D_a)),train_size = int(0.85*len(Datos3D_a)))
Training_3D_B, Testing_3D_B = model_selection.train_test_split(Datos3D_b,test_size = int(0.15*len(Datos3D_b)),train_size = int(0.85*len(Datos3D_b)))

#%% Punto 2b datos 3D
#Grafique el conjunto de entrenamiento con un color para cada clase.
#"""
fig = plt.figure(dpi=600)
Ax = Axes3D(fig)
Ax.scatter(Training_3D_A[:,0],Training_3D_A[:,1],Training_3D_A[:,2],c='red',label = 'Datos 3D A')
Ax.scatter(Training_3D_B[:,0],Training_3D_B[:,1],Training_3D_B[:,2],c='blue',label = 'Datos 3D B')
Ax.legend()
Ax.grid()

#"""

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

Verosimilitud_A=DatosCorrectos_A*Prob_A_Priori_Datos_A/(DatosCorrectos_A*Prob_A_Priori_Datos_A+DatosCorrectos_B*Prob_A_Priori_Datos_B)
Verosimilitud_B=DatosCorrectos_B*Prob_A_Priori_Datos_B/(DatosCorrectos_A*Prob_A_Priori_Datos_A+DatosCorrectos_B*Prob_A_Priori_Datos_B)


### Punto 2c ii
# Grafique la función de verosimilitud de cada clase en un mismo gráfico.

#Clase X
Eje_X_A=np.arange(u_Datos3D_a[0]-3*K_Datos3D_A[0,0],u_Datos3D_a[0]+3*K_Datos3D_A[0,0],(6*K_Datos3D_A[0,0])/30)
Eje_X_B=np.arange(u_Datos3D_b[0]-3*K_Datos3D_B[0,0],u_Datos3D_b[0]+3*K_Datos3D_B[0,0],(6*K_Datos3D_B[0,0])/30)
plt.figure(dpi=600)
plt.plot(Eje_X_A, Verosimilitud_A*norm.pdf(Eje_X_A, u_Datos3D_a[0], K_Datos3D_A[0,0]),c='b',label = 'Grafica Verosimilitud Clase X Datos A')
plt.plot(Eje_X_B, Verosimilitud_B*norm.pdf(Eje_X_B, u_Datos3D_b[0], K_Datos3D_B[0,0]),c='r',label = 'Grafica Verosimilitud Clase X Datos B')
plt.legend()
plt.grid()
plt.show()
#Clase Y
Eje_Y_A=np.arange(u_Datos3D_a[1]-3*K_Datos3D_A[1,1],u_Datos3D_a[1]+3*K_Datos3D_A[1,1],(6*K_Datos3D_A[1,1])/30)
Eje_Y_B=np.arange(u_Datos3D_b[1]-3*K_Datos3D_B[1,1],u_Datos3D_b[1]+3*K_Datos3D_B[1,1],(6*K_Datos3D_B[1,1])/30)
plt.plot(Eje_Y_A, Verosimilitud_A*norm.pdf(Eje_Y_A, u_Datos3D_a[1], K_Datos3D_A[1,1]),c='b',label = 'Grafica Verosimilitud Clase Y Datos A')
plt.plot(Eje_Y_B, Verosimilitud_B*norm.pdf(Eje_Y_B, u_Datos3D_b[1], K_Datos3D_B[1,1]),c='r',label = 'Grafica Verosimilitud Clase Y Datos B')
plt.legend()
plt.grid()
plt.show()
#Clase Z
Eje_Z_A=np.arange(u_Datos3D_a[2]-3*K_Datos3D_A[2,2],u_Datos3D_a[2]+3*K_Datos3D_A[2,2],(6*K_Datos3D_A[2,2])/30)
Eje_Z_B=np.arange(u_Datos3D_b[2]-3*K_Datos3D_B[2,2],u_Datos3D_b[2]+3*K_Datos3D_B[2,2],(6*K_Datos3D_B[2,2])/30)
plt.plot(Eje_Z_A, Verosimilitud_A*norm.pdf(Eje_Z_A, u_Datos3D_a[2], K_Datos3D_A[2,2]),c='b',label = 'Grafica Verosimilitud Clase Z Datos A')
plt.plot(Eje_Z_B, Verosimilitud_B*norm.pdf(Eje_Z_B, u_Datos3D_b[2], K_Datos3D_B[2,2]),c='r',label = 'Grafica Verosimilitud Clase Z Datos B')
plt.legend()
plt.grid()
plt.show()


### Punto 2c iii
# Grafique la clasificación realizada sobre el conjunto de prueba
#"""
Datos_AX=[]
Datos_AY=[]
Datos_AZ=[]
Datos_BX=[]
Datos_BY=[]
Datos_BZ=[]

for i in range(len(Y_Bayessiano)):
    if Y_Bayessiano[i] == 0: 
        Datos_AX.append(Testing[i,0]) #Grupo A Eje X
        Datos_AY.append(Testing[i,1]) #Grupo A Eje Y
        Datos_AZ.append(Testing[i,2]) #Grupo A Eje Z
    elif Y_Bayessiano[i] == 1: 
        Datos_BX.append(Testing[i,0]) #Grupo B Eje X
        Datos_BY.append(Testing[i,1]) #Grupo B Eje Y
        Datos_BZ.append(Testing[i,2]) #Grupo B Eje Z

fig = plt.figure()
Ax = Axes3D(fig)
Ax.scatter(Datos_AX,Datos_AY,Datos_AZ,c='red',label = 'Datos 3D A (Datos de Prueba)')
Ax.scatter(Datos_BX,Datos_BY,Datos_BZ,c='blue',label = 'Datos 3D B (Datos de Prueba)')
Ax.legend()
Ax.grid()
#"""
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

Verosimilitud_A_Naive=DatosCorrectos_A_Naive*Prob_A_Priori_Datos_A/(DatosCorrectos_A_Naive*Prob_A_Priori_Datos_A+DatosCorrectos_B_Naive*Prob_A_Priori_Datos_B)
Verosimilitud_B_Naive=DatosCorrectos_B_Naive*Prob_A_Priori_Datos_B/(DatosCorrectos_A_Naive*Prob_A_Priori_Datos_A+DatosCorrectos_B_Naive*Prob_A_Priori_Datos_B)

### Punto 2d ii
# Grafique las funciones de verosimilitud de cada característica para cada clase en un mismo gráfico.

#Clase X
Eje_X_A=np.arange(u_Datos3D_a[0]-3*K_Datos3D_A[0,0],u_Datos3D_a[0]+3*K_Datos3D_A[0,0],(6*K_Datos3D_A[0,0])/30)
Eje_X_B=np.arange(u_Datos3D_b[0]-3*K_Datos3D_B[0,0],u_Datos3D_b[0]+3*K_Datos3D_B[0,0],(6*K_Datos3D_B[0,0])/30)
plt.figure(dpi=600)
plt.plot(Eje_X_A, Verosimilitud_A_Naive*norm.pdf(Eje_X_A, u_Datos3D_a[0], K_Datos3D_A[0,0]),c='b',label = 'Grafica Verosimilitud Naive Clase X Datos A')
plt.plot(Eje_X_B, Verosimilitud_B_Naive*norm.pdf(Eje_X_B, u_Datos3D_b[0], K_Datos3D_B[0,0]),c='r',label = 'Grafica Verosimilitud Naive Clase X Datos B')
plt.legend()
plt.grid()
plt.show()
#Clase Y
Eje_Y_A=np.arange(u_Datos3D_a[1]-3*K_Datos3D_A[1,1],u_Datos3D_a[1]+3*K_Datos3D_A[1,1],(6*K_Datos3D_A[1,1])/30)
Eje_Y_B=np.arange(u_Datos3D_b[1]-3*K_Datos3D_B[1,1],u_Datos3D_b[1]+3*K_Datos3D_B[1,1],(6*K_Datos3D_B[1,1])/30)
plt.plot(Eje_Y_A, Verosimilitud_A_Naive*norm.pdf(Eje_Y_A, u_Datos3D_a[1], K_Datos3D_A[1,1]),c='b',label = 'Grafica Verosimilitud Naive Clase Y Datos A')
plt.plot(Eje_Y_B, Verosimilitud_B_Naive*norm.pdf(Eje_Y_B, u_Datos3D_b[1], K_Datos3D_B[1,1]),c='r',label = 'Grafica Verosimilitud Naive Clase Y Datos B')
plt.legend()
plt.grid()
plt.show()
#Clase Z
Eje_Z_A=np.arange(u_Datos3D_a[2]-3*K_Datos3D_A[2,2],u_Datos3D_a[2]+3*K_Datos3D_A[2,2],(6*K_Datos3D_A[2,2])/30)
Eje_Z_B=np.arange(u_Datos3D_b[2]-3*K_Datos3D_B[2,2],u_Datos3D_b[2]+3*K_Datos3D_B[2,2],(6*K_Datos3D_B[2,2])/30)
plt.plot(Eje_Z_A, Verosimilitud_A_Naive*norm.pdf(Eje_Z_A, u_Datos3D_a[2], K_Datos3D_A[2,2]),c='b',label = 'Grafica Verosimilitud Naive Clase Z Datos A')
plt.plot(Eje_Z_B, Verosimilitud_B_Naive*norm.pdf(Eje_Z_B, u_Datos3D_b[2], K_Datos3D_B[2,2]),c='r',label = 'Grafica Verosimilitud Naive Clase Z Datos B')
plt.legend()
plt.grid()
plt.show()


### Punto 2d iii
# Grafique la clasificación realizada sobre el conjunto de prueba.

#"""

Datos_AX_N=[]
Datos_AY_N=[]
Datos_AZ_N=[]
Datos_BX_N=[]
Datos_BY_N=[]
Datos_BZ_N=[]

for i in range(len(Y_Bayessiano_Naive)):
    if Y_Bayessiano_Naive[i] == 0: 
        Datos_AX_N.append(Testing[i,0]) #Grupo A Eje X
        Datos_AY_N.append(Testing[i,1]) #Grupo A Eje Y
        Datos_AZ_N.append(Testing[i,2]) #Grupo A Eje Z
    elif Y_Bayessiano_Naive[i] == 1: 
        Datos_BX_N.append(Testing[i,0]) #Grupo B Eje X
        Datos_BY_N.append(Testing[i,1]) #Grupo B Eje Y
        Datos_BZ_N.append(Testing[i,2]) #Grupo B Eje Z

fig = plt.figure()
Ax = Axes3D(fig)
Ax.scatter(Datos_AX_N,Datos_AY_N,Datos_AZ_N,c='red',label = 'Datos 3D A (Datos de Prueba)')
Ax.scatter(Datos_BX_N,Datos_BY_N,Datos_BZ_N,c='blue',label = 'Datos 3D B (Datos de Prueba)')
Ax.legend()
Ax.grid()
#"""

### Punto 2d iv
# Determine el error de clasificación sobre el conjunto de prueba.

Error_Bayes_Naive=1-DatosBuenosNaive/len(Testing)
print("El error de bayes naive es: ",round(100*Error_Bayes_Naive,2),"%")










