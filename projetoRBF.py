import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.signal import butter, sosfiltfilt

data = np.loadtxt('dados_01.dat')
data2 = np.loadtxt('dados_02.dat')

t_treino = data[:,0]
u_treino = data[:,1]
y_treino = data[:,2]

plt.figure(figsize = (14,4))
plt.plot(t_treino,y_treino, lw = 2, alpha = .5, label = 'dados')
plt.legend(loc = 'upper right', bbox_to_anchor = (.98,.99))
plt.show()

N_treino = len(y_treino)

y1_treino = np.concatenate(([0.0, 0.0], y_treino)) 
u1_treino = np.concatenate(([0.0], u_treino)) 

x0 = y1_treino[1 : N_treino + 1] # Corresponde a y[n-1]
x1 = y1_treino[0 : N_treino]     # Corresponde a y[n-2]
x2 = u1_treino[1 : N_treino + 1] # Corresponde a u[n]
x3 = u1_treino[0 : N_treino]     # Corresponde a u[n-1]

sigma = 3.29
centros = np.linspace(y1_treino.max(),y1_treino.min(), 4)

g0 = np.exp(- (1 / (2 * sigma**2)) * (x0 - centros[0])**2)
g1 = np.exp(- (1 / (2 * sigma**2)) * (x1 - centros[1])**2)
g2 = np.exp(- (1 / (2 * sigma**2)) * (x2 - centros[2])**2)
g3 = np.exp(- (1 / (2 * sigma**2)) * (x3 - centros[3])**2)

Phi = np.column_stack((g0, g1, g2, g3))
w = np.linalg.lstsq(Phi, y_treino, rcond=None)[0]

cg0, cg1, cg2, cg3 = w

f_treino = Phi @ w  

plt.figure(figsize=(14, 6))
plt.plot(t_treino, y_treino, "b.", label="Dados Originais", alpha=0.6)
plt.plot(t_treino, f_treino, "--r", label="Ajuste RBF", linewidth=2)

fit_label = (
    'Ajuste: 'f'y[n] = {cg0:.4f}y[n-1] + {cg1:.4f}y[n-2] + {cg2:.4f}u[n] + {cg3:.4f}u[n-1]'
)
plt.text(0.015, 0.96, fit_label, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))

plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Ajuste de Curva por RBF")
plt.legend(loc='upper right')
plt.show()

print(f"w0 =  {cg0:.4f}")
print(f"w1 =  {cg1:.4f}")
print(f"w2 =  {cg2:.4f}")
print(f"w3 =  {cg3:.4f}")

p = np.poly1d( np.polyfit(y_treino, f_treino, 1) )
t = np.linspace(-3, 3, 500)
#
EMQ_treino = np.mean((y_treino - f_treino)**2)
#
print(f"EMQ de Treinamento: {EMQ_treino:.6f}")
plt.figure(figsize = (14,4))
plt.suptitle("Erro de treinamento", fontsize=18, y=0.95)
plt.plot(y_treino, f_treino, 'o', t, p(t), '-')
plt.text(2.5, 2.7, 'EMQ = %.4f' %EMQ_treino)
#plt.text(2.5, 2, 'STD = %.4f' %STD)
plt.grid(True)
plt.show()

t_teste = data2[:,0]
u_teste = data2[:,1]
y_teste = data2[:,2]

plt.figure(figsize = (14,4))
plt.plot(t_teste,y_teste, lw = 2, alpha = .5, label = 'dados2')
plt.legend(loc = 'upper right', bbox_to_anchor = (.98,.99))
plt.show()

y1_teste = np.concatenate(([0.0, 0.0], y_teste))
u1_teste = np.concatenate(([0.0], u_teste))

N_teste = len(y_teste)

x0_teste = y1_teste[1 : N_teste + 1] # Corresponde a y[n-1]
x1_teste = y1_teste[0 : N_teste]     # Corresponde a y[n-2]
x2_teste = u1_teste[1 : N_teste + 1] # Corresponde a u[n]
x3_teste = u1_teste[0 : N_teste]     # Corresponde a u[n-1]

g0_teste = np.exp(- (1 / (2 * sigma**2)) * (x0_teste - centros[0])**2)
g1_teste = np.exp(- (1 / (2 * sigma**2)) * (x1_teste - centros[1])**2)
g2_teste = np.exp(- (1 / (2 * sigma**2)) * (x2_teste - centros[2])**2)
g3_teste = np.exp(- (1 / (2 * sigma**2)) * (x3_teste - centros[3])**2)

Phi_teste = np.column_stack((g0_teste, g1_teste, g2_teste, g3_teste))

f_teste = Phi_teste @ w

plt.figure(figsize=(14, 6))
plt.plot(t_teste, y_teste, "b.", label="Dados Originais de Teste", alpha=0.6)
plt.plot(t_teste, f_teste, "--r", label="Previsão RBF (Teste)", linewidth=2)

plt.text(0.015, 0.96, "Modelo RBF aplicado aos dados de teste", transform=plt.gca().transAxes,
         fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))

plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Previsão do Modelo RBF em Dados de Teste")
plt.legend(loc='upper right')
plt.show()

p = np.poly1d( np.polyfit(y_teste, f_teste, 1) )
t = np.linspace(np.min(y_teste), np.max(y_teste), 500)
#
EMQ_teste = np.mean((y_teste - f_teste)**2)
#
print(f"EMQ de Teste: {EMQ_teste:.6f}")
plt.figure(figsize=(14, 4))
plt.suptitle("EMQ de Teste", fontsize=18, y=0.95)
plt.plot(y_teste, f_teste, 'o')
plt.plot(t, p(t), '-')
plt.text(0.95, 0.05, f'EMQ = {EMQ_teste:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

N_degrau = 200
t_degrau = np.arange(N_degrau)
u_degrau = np.ones(N_degrau)
y_degrau = np.zeros(N_degrau)

for n in range(2, N_degrau):
    x0_step = y_degrau[n - 1]
    x1_step = y_degrau[n - 2]
    x2_step = u_degrau[n]
    x3_step = u_degrau[n - 1]

    g0_step = np.exp(- (1 / (2 * sigma**2)) * (x0_step - centros[0])**2)
    g1_step = np.exp(- (1 / (2 * sigma**2)) * (x1_step - centros[1])**2)
    g2_step = np.exp(- (1 / (2 * sigma**2)) * (x2_step - centros[2])**2)
    g3_step = np.exp(- (1 / (2 * sigma**2)) * (x3_step - centros[3])**2)

    phi_step = np.array([g0_step, g1_step, g2_step, g3_step])
  
    y_degrau[n] = phi_step @ w

plt.figure(figsize=(14, 6))
plt.plot(t_degrau, y_degrau, "-r", label="Resposta ao Degrau do Modelo RBF", linewidth=2)
plt.title("Resposta ao Degrau do Modelo RBF")
plt.xlabel("Tempo (amostras)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
