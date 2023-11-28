# -*- coding: utf-8 -*-

from scipy.stats.qmc import Sobol
import random
import math as mt
import numpy as np

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

    #Inicializa amostrador aleatório
    amostrador = Sobol(d=1,scramble=False)
    
    #Gera números quasi-aleatórios
    amostra = amostrador.random_base2(mt.ceil(mt.log2(95)))

    #Retorna estimativa calculada
    return np.mean(list(map(f,amostra)))

#Monte Carlo Acertou ou Errou
def hit_or_miss(Seed = None):
    random.seed(Seed)

    #Inicializa amostrador aleatório
    amostrador = Sobol(d=1,scramble=False)
    
    #Gera números quasi-aleatórios
    amostra1 = amostrador.random_base2(mt.ceil(mt.log2(329)))
    amostra2 = amostrador.random_base2(mt.ceil(mt.log2(329)))
    
    #Retorna estimativa calculada
    return np.mean(amostra2<list(map(f,amostra1)))

#Monte Carlo Amostragem por Importância
def importance_sampling(Seed = None):
    random.seed(Seed)
    
    #Inicializa amostrador aleatório
    amostrador = Sobol(d=1,scramble=False)
    
    #Gera números quasi-aleatórios
    amostra = amostrador.random_base2(mt.ceil(mt.log2(70)))
    
    #Gera valores exponencialmente distribuidos com lambda = 1 ,a partir da amostra anterior
    amostra = list(map(lambda x: mt.log(1-x)*(-1),amostra))
    
    #Retorna estimativa calculada
    return np.mean(list(map(lambda x: f(x)/mt.exp(-x) if x<=1 else 0 ,amostra)))

#Monte Carlo por Variável de Controle
def control_variate(Seed = None):
    random.seed(Seed)
    
    #Define o polinômio phi
    def phi(x):
        return 0.99-0.48*x+0.06*x**2

    #Retorna a integral do polinômio 0.99-0.48x+0.06x^2
    gamma_l = 0.99-(0.48)/2+(0.06)/3
    
    #Inicializa amostrador aleatório
    amostrador = Sobol(d=1,scramble=False)
    
    #Gera números quasi-aleatórios
    amostra = amostrador.random_base2(mt.ceil(mt.log2(3)))
    
    #Retorna estimativa calculada    
    return np.mean(list(map(lambda x: f(x)-phi(x)+gamma_l,amostra)))

if __name__ == "__main__":
    #Coloque seus testes aqui
    print(crude())
    print(hit_or_miss())
    print(importance_sampling())
    print(control_variate())
