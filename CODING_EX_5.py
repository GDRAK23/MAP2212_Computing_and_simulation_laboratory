#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Estimador:
    
    #Inicialização da classe e armazenamento dos parâmetros da Dir(x+y-1)
    def __init__(self,x,y):
        
        import numpy as np
        import math as mt
        
        self.vetor_x = np.array(x)
        self.vetor_y = np.array(y) 
        
        #Método que gera n variáveis aleatórias distribuídas segundo Dir(x+y-1) através do Método Markov Chain Monte Carlo 
        def MCMC(n):
            
            #Define a matriz de covariância a ser usada no processo
            S=np.array([ ((self.vetor_x+self.vetor_y-1)[i]*(sum((self.vetor_x+self.vetor_y-1))-(self.vetor_x+self.vetor_y-1)[i]))/
                        (sum((self.vetor_x+self.vetor_y-1))**2*(sum((self.vetor_x+self.vetor_y-1))+1)) if i==j 
                        else -((self.vetor_x+self.vetor_y-1)[i]*(self.vetor_x+self.vetor_y-1)[j])/
                        (sum((self.vetor_x+self.vetor_y-1))**2*(sum((self.vetor_x+self.vetor_y-1))+1)) 
                        for i in range(3) for j in range(3) ]).reshape(3,3)
    
            #Função de densidade de probabilidade de Dir(x+y-1)
            def g(theta,x,y):
                return np.prod([pow(theta[i],x[i]+y[i]-1) for i in range(len(x))])
            
            #Definindo ponto inicial x0
            x_=np.array([0.2,0.5,0.3])
            
            #Lista que armazena os pontos gerados
            p=[]
            
            #Loop que executa o algoritmo de aceitação-rejeição com probabilidade de aceitação de Metropolis-Hastings
            for i in range(n):

                #Calcula xj
                x_prop=x_+np.random.multivariate_normal([0,0,0],S)

                #Testa a aceitação e a adequação de xj ao simplex S3 bem como sua positividade e limitação superior por 1
                if np.random.uniform()<float(min(1,g(x_prop,x,y)/g(x_,x,y))) and sum(x_prop>0)==3 and sum(x_prop<1)==3 :
                
                    #Aceito o novo xj proposto e armazeno
                    x_ = x_prop
                    p.append(x_)
            
            #Descarta período de burn-in do processo 
            p=p[:len(p)-200]
            #Retorno uma lista de valores g(θ) devidamente ajustado à escala
            return mt.gamma(sum(x+y))/np.prod([mt.gamma(x+y) for x,y in zip(x,y) ]) * np.array([np.prod(p) for p in list(map(lambda theta:[pow(theta[i],x[i]+y[i]-1) for i in range(len(x))],p))])
        
        #Realiza simulação com n=10.000~Dir(x+y-1) pontos através do método MCMC, para achar uma aproximação do sup f(θ)
        self.supf=max(MCMC(10000))
        #Define o número de bins
        self.bins=np.arange(start=0,stop=self.supf,step=self.supf/154)
        
        #Simulando n pontos~Dir(x+y-1) através do método MCMC
        dirch_pontos=MCMC(192080)
        
        #Calculando a proporção de pontos que cairam nos bins
        self.prop=[sum(np.digitize(dirch_pontos,self.bins)==b)/len(dirch_pontos) for b in np.unique(np.digitize(dirch_pontos, self.bins))]
    
    def U(self,v):

        import numpy as np
        
        #Retorna 1 se v maior que sup f(θ)
        if v>self.supf:
            return 1
        #Retorna 0 se v=0
        elif v==0:
            return 0
        else:
            #Obtem o extremo interior do bin
            pos = np.digitize(v,self.bins)-1
            #Calcula a base do retangulo [inf bin_i,v] e multiplica pela proporção de pontos que caíram no bin
            val = (v-self.bins[pos])*self.prop[pos]
        #Retorna a estimativa U(v)
        
        return sum(self.prop[0:pos])+val


import time

agora=time.time()
estimativa = Estimador(x=[1,2,3],y=[4,5,6])

[(i,estimativa.U(i)) for i in range(19)]

depois=time.time()

print((depois-agora))
print((depois-agora)/60)
