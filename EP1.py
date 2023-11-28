#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
#Escreva seu nome e numero USP
INFO = {7199162:"Guilherme Augusto Martins"}
def estima_pi(Seed = None):
    
    random.seed(Seed)
    #random.random() gera um numero com distribuicao uniforme em (0,1)
    """
    Esta funcao deve retornar a sua estimativa para o valor de PI
    Escreva o seu codigo nas proximas linhas
    """
    
    def Monte_Carlo(n):
        count=0;
        for i in range(0,n):
            #Gera número aleatório entre 0 e 1
            x = random.random();
            y = random.random();
            #Se a norma do ponto gerado for menor que um incrementa o contador
            if ((x**2+y**2)**(1/2))<1:
                count=count+1;
            
        #Multiplica por 4 e imprime o valor
        pi = (count/n);
        
        return pi
    
    #Realizando de uma simulação inicial para o valor de pi/4
    lista=[]
    import numpy as np
    for i in range(50):
        lista.append(Monte_Carlo(1000000));
    estim_ini=np.mean(lista)
    
    #Através da fórmula para cálculo de n com um intervalo de confiança de 95% baseado na N(0,1) definimos o n adequado
    z_gama = 1.96
    S_2 = ( estim_ini*(1-estim_ini) )
    epsilon = 0.0005*estim_ini
    
    n = ((z_gama**2) * (S_2)) / ((epsilon)**2)
    
    #Recalculamos a estimativa de pi com o n obtido
    estim_final = 4*Monte_Carlo(int(n))
    
    return estim_final

pi=estima_pi()
