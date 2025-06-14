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

y1_treino = np.concatenate(([0.0, 0.0], y_treino)) # agora y = [0,0,(y original)]
u1_treino = np.concatenate(([0.0], u_treino)) # e u = [0,u(original)]

g0 = y1_treino[1 : N_treino + 1] # Corresponde a y[n-1]
g1 = y1_treino[0 : N_treino]     # Corresponde a y[n-2]
g2 = u1_treino[1 : N_treino + 1] # Corresponde a u[n]
g3 = u1_treino[0 : N_treino]     # Corresponde a u[n-1]

A = np.array([
    [g0.T @ g0, g0.T @ g1, g0.T @ g2, g0.T @ g3],
    [g1.T @ g0, g1.T @ g1, g1.T @ g2, g1.T @ g3],
    [g2.T @ g0, g2.T @ g1, g2.T @ g2, g2.T @ g3],
    [g3.T @ g0, g3.T @ g1, g3.T @ g2, g3.T @ g3]
])

b = np.array([
    [g0.T @ y_treino],
    [g1.T @ y_treino],
    [g2.T @ y_treino],
    [g3.T @ y_treino]
])

w = la.solve(A, b)

cg0, cg1, cg2, cg3 = w.flatten()

f_treino = np.zeros_like(y_treino)

for n in range(N_treino):
    f_treino[n] = cg0 * y1_treino[n+1] + cg1 * y1_treino[n] + cg2 * u1_treino[n+1] + cg3 * u1_treino[n]

plt.figure(figsize=(14, 6))
plt.plot(t_treino, y_treino, "b.", label="Dados Originais", alpha=0.6)
plt.plot(t_treino, f_treino, "--r", label="Ajuste MMQ", linewidth=2)

fit_label = (
    'Ajuste: 'f'y[n] = {cg0:.4f}y[n-1] + {cg1:.4f}y[n-2] + {cg2:.4f}u[n] + {cg3:.4f}u[n-1]'
)
plt.text(0.015, 0.96, fit_label, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))

plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Ajuste de Curva por Mínimos Quadrados (MMQ)")
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

f_teste = np.zeros_like(y_teste)
for n in range(N_teste):
    f_teste[n] = cg0 * y1_teste[n+1] + cg1 * y1_teste[n] + cg2 * u1_teste[n+1] + cg3 * u1_teste[n]

plt.figure(figsize=(14, 6))
plt.plot(t_teste, y_teste, "b.", label="Dados Originais", alpha=0.6)
plt.plot(t_teste, f_teste, "--r", label="Ajuste MMQ", linewidth=2)

fit_label = (
    'Ajuste: 'f'y[n] = {cg0:.4f}y[n-1] + {cg1:.4f}y[n-2] + {cg2:.4f}u[n] + {cg3:.4f}u[n-1]'
)
plt.text(0.015, 0.96, fit_label, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))

plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Ajuste de Curva por Mínimos Quadrados (MMQ)")
plt.legend(loc='upper right')
plt.show()

p = np.poly1d( np.polyfit(y_teste, f_teste, 1) )
t = np.linspace(-3, 3, 500)
#
EMQ_teste = np.mean((y_teste - f_teste)**2)
#
print(f"EMQ de Teste: {EMQ_teste:.6f}")
plt.figure(figsize=(14, 4))
plt.suptitle("EMQ de Teste", fontsize=18, y=0.95)
plt.plot(y_teste, f_teste, 'o', label='Dados')
plt.plot(t, p(t), '-')
plt.text(0.95 * max(y_teste), 0.95 * max(f_teste), f'EMQ = {EMQ_teste:.4f}')
plt.xlabel('y_teste')
plt.ylabel('f_teste')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
