# -*- coding: utf-8 -*-

import random

#Escreva seu nome e numero USP
INFO = {7199162:"Guilherme Augusto Martins"}
A = 0.48513472  # A = 0.rg
B = 0.36450663812  # B = 0.cpf

#Função a integrar
def f(x):
    #importa biblioteca de funções matemáticas
    import math as mt
    return mt.exp(-A*x)*mt.cos(B*x)

#Monte Carlo Cru
def crude(Seed = None):
    random.seed(Seed)

    #Define n adequado
    n=95
    #Inicializa estimador
    gamma_hat=0
    for i in range(n):
        #Gera xi ~ U[0,1] e calcula f(xi)*(1/n)
        gamma_hat += f(random.random())*(1/n)
    return gamma_hat

#Monte Carlo Acertou ou Errou
def hit_or_miss(Seed = None):
    random.seed(Seed)

    #Define n adequado
    n=329
    #Inicializa estimador
    gamma_hat=0
    for i in range(n):
        #Gera xi , yi ~ U[0,1]
        x=random.random()
        y=random.random()
        #Se estiver abaixo do gráfico da função incrementa o estimador
        if y<f(x):
            gamma_hat+=1*(1/n)
    return gamma_hat

def importance_sampling(Seed = None):
    random.seed(Seed)
    
    from scipy.stats import expon
    import math as mt
    
    #Define n adequado
    n=70
    #Inicializa estimador  
    gamma_hat=0
    for i in range(n):
        #Gera xi conforme g(x)
        x=expon.rvs()
        #Se estiver entre 0 e 1
        if x<=1:
            #Incrementa a estimativa
            gamma_hat += (f(x)/mt.exp(-x))*(1/n)
        
    return gamma_hat

#Monte Carlo por Variável de Controle
def control_variate(Seed = None):
    random.seed(Seed)
    
    #Define o polinômio phi
    def phi(x):
        return 0.99-0.48*x+0.06*x**2

    #Retorna a integral do polinômio 0.99-0.48x+0.06x^2
    gamma_l = 0.99-(0.48)/2+(0.06)/3
    
    #Define n adequado
    n=3
    #Inicializa estimador
    gamma_hat=0
    for i in range(n):
        #Gera xi ~ U[0,1]
        x= random.random()
        #Incrementa estimador
        gamma_hat+=(1/n)*(f(x)-phi(x)+gamma_l)

    return gamma_hat

if __name__ == "__main__":
    #Coloque seus testes aqui
    print(crude())
    print(hit_or_miss())
    print(importance_sampling())
    print(control_variate())
