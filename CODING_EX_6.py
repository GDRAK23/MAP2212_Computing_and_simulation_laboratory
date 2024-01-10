#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Classe que calcula uma aproximação da função verdade de um potencial ~ Dir(x+y-1)
class Estimador:
    
    #Inicialização da classe e armazenamento dos parâmetros da Dir(x+y-1)
    def __init__(self,x,y):
    
        
        import numpy as np
        
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
            return np.array([np.prod(p) for p in list(map(lambda theta:[pow(theta[i],x[i]+y[i]-1) for i in range(len(x))],p))])
        
        #Realiza simulação com n=10.000~Dir(x+y-1) pontos através do método MCMC, para achar uma aproximação do sup f(θ)
        self.supf=max(MCMC(10000))
  
        #Define o número de bins
        self.bins=np.arange(start=0,stop=self.supf,step=self.supf/144)
        
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
    
def main():
    #Importa bibliotecas utilizadas no processo
    import numpy as np
    import math as mt
    from scipy.optimize import minimize
    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('always',category=UserWarning)
    
    #Define os quantis e a função de valor epistêmico padronizado
    def QQ(t,h,z):
        from scipy.stats.distributions import chi2
        return chi2.cdf(chi2.ppf(z,t),t*h)
    
    def sev(t,h,ev):
        return 1-QQ(t,h,1-ev)
    
    #Gera os pontos conforme disponível em C.A.B.Pereira, J.M.Stern, (1999). Evidence and Credibility: Full Bayesian Significance Test for Precise Hypotheses. Entropy Journal, 1, 69-80. 
    x=np.concatenate((np.array([[1,0,i] for i in range(2,19)]),np.array([[5,0,i] for i in range(0,11)]),np.array([[9,0,i] for i in range(0,8)])),axis=0)
    x[:,1] = 20-x[:,0]-x[:,2]
    p = np.concatenate((np.concatenate((x,x),axis=0),np.concatenate((np.array([[0,0,0] for i in range(len(x))]),np.array([[1,1,1] for i in range(len(x))])),axis=0)), axis=1)
    
    #Executa o teste da hipótese de equilíbrio de Hardy-Weinberg para os pontos gerados acima
    for i in range(len(p)):
        
        #Obtém valores dos vetores x e y, bem como o ponto inicial do método de otimização
        x=p[i][0:3]
        y=p[i][3:6]
        theta0=(0.3333,0.3333,0.3333)
        
        #Define a função potencial a ser maximizada sob a hipótese (restrição) e as restrições de domínio
        g1 = lambda theta : -1*((pow(theta[0],x[0]+y[0]-1))*(pow(theta[1],x[1]+y[1]-1))*(pow(theta[2],x[2]+y[2]-1)))
        cons = ({'type': 'eq', 'fun': lambda theta:  theta[2] - pow(1-mt.sqrt(theta[0]),2)   },{'type': 'eq', 'fun': lambda theta:  theta[0]+theta[1]+theta[2]-1   })
        bnds=((0,1),(0,1),(0,1))
        res = minimize(g1, theta0,bounds=bnds,constraints=cons)
        
        #Função potencial na sua forma original
        g2 = lambda theta : ((pow(theta[0],x[0]+y[0]-1))*(pow(theta[1],x[1]+y[1]-1))*(pow(theta[2],x[2]+y[2]-1)))
        
        #Valor da função potencial no seu ponto de máximo
        s=(g2)(res.x)
        #dim(Θ)
        t=3
        #dim(H)
        h=2
        #Calcula a função verdade em Θ*
        estimativa = Estimador(x,y)
        #Toma a decisão de rejeitar ou não a hipótese
        if sev(t,h,estimativa.U(s))<0.05: 
            decisao="sim" 
        else: 
            decisao="não"
        #Exibde valores calculados
        print("x1= "+str(x[0])+' ,'+"x3= "+str(x[2])+' ,Y= '+str(y)+' ,H='+decisao,' ,θ*='+str(res.x)+' ,ev(H|X)='+str(estimativa.U(s))+' ,sev(H|X)='+str(sev(t,h,estimativa.U(s))))
  
if __name__ == '__main__':
    main()
