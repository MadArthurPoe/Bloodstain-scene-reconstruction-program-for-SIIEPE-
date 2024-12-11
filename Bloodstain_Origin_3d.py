import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar


#######################################################
### This code was described in brazilian portuguese ###
#######################################################


#### Constantes ####
rho = 1006  # Densidade (kg/m³)
gamma = 0.0518  # Tensão superficial (N/m)
mi = 0.0013  # Viscosidade (Pa.s)
g = 9.81  # Aceleração da gravidade em m/s^2

#### Weber e Reynolds como funcao ####
def weber(rho, v0, d0, gamma):
    return (rho * v0**2 * d0) / gamma

def reynolds(rho, v0, d0, mi):
    return (rho * v0 * d0) / mi

#### Equacao principal em partes ####
### Constantes variam para cada fluido 
def equacao(v0, d_cmax, d0, rho, gamma, mi): 
    We = weber(rho, v0, d0, gamma)
    Re = reynolds(rho, v0, d0, mi)
    termo_exp = math.exp(-1.45 * We**0.18 * Re**-0.18)
    return abs(d_cmax/d0 - (1 + 0.55 * We**0.55 * termo_exp))

#### Calculo e resolve problema de otimizacao, limitando a funcao ####
def calcular_v0(d_cmax, d0, rho, gamma, mi):
    resultado = minimize_scalar(equacao, bounds=(0.1, 100.0), args=(d_cmax, d0, rho, gamma, mi), method='bounded')
    return resultado.x

#### Calcula a incerteza da medida (Apenas se necessario!!) ####
def calcular_com_incerteza(d_cmax, d_cmax_unc, d0, d0_unc, rho, gamma, mi):
    if d_cmax_unc is not None:
        d_cmax = (d_cmax + d_cmax_unc) / 2
    if d0_unc is not None:
        d0 = (d0 + d0_unc) / 2
    
    v0 = calcular_v0(d_cmax, d0, rho, gamma, mi)
    return v0

#### Dispositivo para calcular os valores das velocidades de impacto a partir de matriz de dados ####
def calcular_velocidades_matriz(dados):
    velocidades = []
    for linha in dados:
        d_cmax, d_cmax_unc, d0, d0_unc = linha
        v0 = calcular_com_incerteza(d_cmax, d_cmax_unc, d0, d0_unc, rho, gamma, mi)
        velocidades.append(v0)
    return velocidades

#### Plot 3D ####
def plotar_trajetoria(ax, v, theta, phi, origem, indice):
    # Componentes da velocidade
    vx = v * np.cos(theta) * np.cos(phi)
    vy = v * np.cos(theta) * np.sin(phi)
    vz = v * np.sin(theta)

    #### Tempo estimado ####
    t_flight = 1  # Tempo de subida e descida

    #### Gera os tempos de voo ####
    t = np.linspace(0, t_flight, num=500)

    #### Trajetoria ####
    x = origem[0] + vx * t
    y = origem[1] + vy * t
    z = origem[2] + vz * t - 0.5 * g * t**2

    #### Plotar da trajetoria ####
    ax.plot(x, y, z, 'red')  # Todas as trajetórias em vermelho
    
    #### Adiciona valor para a origem respectiva de cada mancha!! ####
    ax.text(origem[0], origem[1], origem[2], str(indice), color='black', fontsize=12, ha='right')

#### Funcao para adicionar um disco em 3D ####
def adicionar_disco(ax, centro, raio, cor):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = centro[0] + raio * np.outer(np.cos(u), np.sin(v))
    y = centro[1] + raio * np.outer(np.sin(u), np.sin(v))
    z = centro[2] + np.zeros_like(x)  # Disco plano paralelo ao eixo XY
    ax.plot_surface(x, y, z, color=cor, alpha=0.5)

#### Matriz de dados (sem rho, gamma, mi pois sao constantes para todas as manchas) ####
### Esses dados sao respectivos para as unidades de medida em paquimetro, coletadas para sua comprovacao, se necessario, leia o resumo expandido que contem a descricao e limitacoes do codigo. 
dados = np.array([
    [0.003, 0.003, 0.001, 0.001], #1
    [0.0031, 0.0031, 0.001, 0.001], #2
    [0.0018, 0.0018, 0.00065, 0.00065], #3
    [0.0046, 0.0046, 0.0006, 0.0006], #4
    [0.0054, 0.0054, 0.001, 0.001], #5
    [0.00685, 0.00685, 0.0008, 0.0008], #6
    [0.00315, 0.00315, 0.0006, 0.0006], #7
    [0.001, 0.0009, 0.0003, 0.0005], #8
    [0.001, 0.001, 0.00044, 0.0005], #9
    #### Adicionar mais linhas conforme necessario ####
])

#### Calculando velocidades para cada conjunto de dados na matriz ####
velocidades = calcular_velocidades_matriz(dados)

#### Dados de entrada adicionais ####
angulos = [69, 73.4, 75.31, 231.340, 239.036,
            236.633, 230.42, 73.56, 68]  ## Lista de angulos em graus
phis = [325, 316, 313, 333, 334, 335, 332, 5, 6]  ## Lista de angulos em graus no plano XY
origens = [(0, 0.2835, 0.67), (0, 0.337, 0.745), (0, 0.38, 0.6335), (0.567, 0, 1.02), 
           (0.534, 0, 1.005), (0.62, 0, 1.116), (0.549, 0, 0.894), (0, 0.14, 0.609), 
           (0, 0.133, 0.727), (0, 0, 0.576)]  ## Lista de origens
coordenada_bolinha = (0.203, 0.157, 0.54)  ## Coordenada da bolinha azul (0.54, 0.157) (Origem real)

#### Pontos de intersecao fornecidos #### Funciona se nao for plot 3D !!!!
### Esses pontos voce vai coletar analiticamente
### Funciona apenas para 2D
intersecoes = np.array([
    [0.2352, 0.157, 0.5961],
    [0.2317, 0.157, 0.57],
    [0.2267, 0.157, 0.531],
    [0.24, 0.157, 0.55],
    [0.2454, 0.157, 0.5589],
    [0.2044, 0.157, 0.5518],
    [0.1910,0.157, 0.53],
    [0.2115, 0.157, 0.5063],
    [0.1988, 00.157, 0.4831],
    [0.21, 00.157, 0.42],
    [0.2211, 00.157, 0.4424],
    [0.2391, 00.157, 0.4751],
    [0.2562, 00.157, 0.5058],
    [0.2674, 00.157, 0.455],
    [0.26, 00.157, 0.4361],
    [0.251, 00.157, 0.4145],
    [0.2311, 00.157, 0.3642],
    #[0.2214, 0, 0.3394],
])

#### Calcular a media das interseções (Apenas 2D) ####
media_intersecoes = np.mean(intersecoes, axis=0)

#### Plotar a reta de referencia e as trajetorias #### (Nao funcional!!)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#### Loop para plotar multiplas trajetorias ####
for i, (v, theta_deg, phi_deg, origem) in enumerate(zip(velocidades, angulos, phis, origens), start=1):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    plotar_trajetoria(ax, v, theta, phi, origem, i)

## Adicionar ponto de intersecao medio
#ax.scatter(media_intersecoes[0], media_intersecoes[1], media_intersecoes[2], color='green', s=400, label='Origem Reconstruída por interseções')

## Adicionar a coordenada da bolinha
### Essa origem eh apenas visual, se quiser, voce pode adicionar como origem suposta, ja que isso foi referente ao uso do experimento
ax.scatter(coordenada_bolinha[0], coordenada_bolinha[1], coordenada_bolinha[2], color='blue', s=400, label='Origem real')

## Adicionar legenda ##
ax.legend()

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Trajetórias do Experimento (Paquimetro) V0.9')
ax.grid(True)
plt.show()

## Calcular o desvio entre as bolinhas ## (2D)
### Esses comandos podem ser relevantes caso voce nao queira parallax
desvio = np.linalg.norm(np.array(media_intersecoes) - np.array(coordenada_bolinha))
desvio_percentual = (desvio / np.linalg.norm(coordenada_bolinha)) * 100
print(f'Desvio: {desvio:.4f} metros')
print(f'Desvio percentual: {desvio_percentual:.2f}%')

desvio, desvio_percentual
