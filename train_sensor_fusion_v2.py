"""
train_sensor_fusion_v2.py — SensorFusion NEXUS Fase 5.4 v2
============================================================
Fix: nn.MultiheadAttention en lugar de nn.Linear para pesos de atencion.

Problema v1: Linear(embed*4, 4) -> gradiente muy diluido, pesos casi fijos.
Fix v2:
  - MultiheadAttention(embed_dim, num_heads=4) con Q=K=V=embeddings
  - Batch ordering correcto para residuo ODE
  - NRMSE objetivo: ALTA (<0.01)

Sistema: pendulo simple (g=9.81, L=1.0, theta0=45 grados)
4 sensores: Visual | Espectral (16 bandas) | IMU | Temperatura
"""
import os, random, math, time, datetime
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SensorFusion v2] Device={DEVICE} | PyTorch {torch.__version__}")

# ── Sistema fisico ─────────────────────────────────────────────────────────────
G, L    = 9.81, 1.0
theta0  = math.pi/4
omega0  = 0.0
T_CICLO = 2.0*math.pi/math.sqrt(G/L)
t_end   = 6*T_CICLO
FPS     = 50
N       = int(t_end * FPS)
t_eval  = np.linspace(0, t_end, N)
dt      = t_eval[1] - t_eval[0]

print(f"[Sistema] g={G} L={L} theta0={math.degrees(theta0):.0f}deg")
print(f"[Dataset] {N} frames  dt={dt:.4f}s  t_end={t_end:.2f}s")

def run_pendulo():
    def fun(t, y): return [y[1], -(G/L)*math.sin(y[0])]
    sol = solve_ivp(fun, (0,t_end), [theta0,omega0], method='LSODA',
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.y.T

sol        = run_pendulo()
theta_true = sol[:,0]
omega_true = sol[:,1]

# ── 4 Sensores ─────────────────────────────────────────────────────────────────
np.random.seed(SEED)

# Sensor 1 — Visual
L_pix = 200.0
px_true = L_pix * np.sin(theta_true)
py_true = L_pix * np.cos(theta_true)
px_obs  = (px_true + np.random.normal(0, 2.0, N)) / L_pix
py_obs  = (py_true + np.random.normal(0, 2.0, N)) / L_pix

# Sensor 2 — Espectral 16 bandas
N_BANDAS = 16
A_spec   = np.random.randn(N_BANDAS, 2)
A_spec   = A_spec / np.linalg.norm(A_spec, axis=1, keepdims=True)
spec_clean = sol @ A_spec.T
spec_std   = spec_clean.std(axis=0, keepdims=True)
spec_obs   = spec_clean + np.random.normal(0, spec_std*0.1, spec_clean.shape)
spec_min   = spec_obs.min(axis=0); spec_max = spec_obs.max(axis=0)
spec_norm  = 2*(spec_obs-spec_min)/(spec_max-spec_min+1e-8)-1

# Sensor 3 — IMU aceleracion
ax_true = -(G/L)*np.sin(theta_true)*L_pix
ay_true =  omega_true**2 * L_pix
ax_obs  = (ax_true + np.random.normal(0, 5.0, N)) / (G*L_pix)
ay_obs  = (ay_true + np.random.normal(0, 5.0, N)) / (G*L_pix)

# Sensor 4 — Temperatura
temp_true = np.abs(omega_true)
temp_obs  = (temp_true + np.random.normal(0, 0.05*temp_true.std(), N))
temp_obs  = (temp_obs - temp_obs.min()) / (temp_obs.max()-temp_obs.min()+1e-8)

# Normalizar tiempo
t_norm = t_eval / t_end

# Dataset
DIM_VIS  = 2; DIM_SPEC = N_BANDAS; DIM_IMU = 2; DIM_TEMP = 1
DIM_TOT  = 1 + DIM_VIS + DIM_SPEC + DIM_IMU + DIM_TEMP

X_all = np.column_stack([
    t_norm,
    px_obs, py_obs,
    spec_norm,
    ax_obs, ay_obs,
    temp_obs
]).astype(np.float32)

Y_all = np.column_stack([theta_true, omega_true]).astype(np.float32)
# Normalizar salida
Y_mean = Y_all.mean(axis=0); Y_std = Y_all.std(axis=0)
Y_norm = (Y_all - Y_mean) / (Y_std + 1e-8)

# Tensores
X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
Y_t = torch.tensor(Y_norm, dtype=torch.float32).to(DEVICE)
t_t = torch.tensor(t_norm, dtype=torch.float32).to(DEVICE)
theta_t = torch.tensor(theta_true, dtype=torch.float32).to(DEVICE)
omega_t = torch.tensor(omega_true, dtype=torch.float32).to(DEVICE)

# ── ARQUITECTURA v2 con MultiheadAttention ────────────────────────────────────
HIDDEN = 128; EMBED = 64; NHEADS = 4

class ModalEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, EMBED),  nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class MultimodalAttentionFusion(nn.Module):
    """
    Fix v2: usa nn.MultiheadAttention con embeddings como secuencia.
    Input:  lista de 4 tensores (N, EMBED)
    Output: tensor fusionado (N, EMBED), pesos de atencion (N, 4)
    """
    def __init__(self, embed_dim, num_heads, n_modalities):
        super().__init__()
        self.n_mod = n_modalities
        # MultiheadAttention: Q=K=V = embeddings apilados como secuencia
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,  # (N, seq, embed)
            dropout=0.0
        )
        # Proyeccion final
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN), nn.Tanh(),
        )

    def forward(self, embeddings):
        # embeddings: lista de n_mod tensores (N, EMBED)
        # Apilar como secuencia: (N, n_mod, EMBED)
        seq = torch.stack(embeddings, dim=1)
        # Self-attention sobre la secuencia de modalidades
        # Cada modalidad "atiende" a todas las otras
        attn_out, attn_weights = self.mha(seq, seq, seq)
        # attn_out:     (N, n_mod, EMBED)
        # attn_weights: (N, n_mod, n_mod)

        # Promedio ponderado = suma sobre modalidades
        fused = attn_out.mean(dim=1)  # (N, EMBED)
        # Pesos de importancia por modalidad (diagonal de attn_weights)
        w = attn_weights.mean(dim=1)  # (N, n_mod) — peso promedio recibido

        return self.proj(fused), w

class SensorFusionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_vis  = ModalEncoder(DIM_VIS)
        self.enc_spec = ModalEncoder(DIM_SPEC)
        self.enc_imu  = ModalEncoder(DIM_IMU)
        self.enc_temp = ModalEncoder(DIM_TEMP)
        self.fusion   = MultimodalAttentionFusion(EMBED, NHEADS, 4)
        self.decoder  = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2)
        )

    def forward(self, x):
        # Separar columnas: [t, vis(2), spec(16), imu(2), temp(1)]
        t    = x[:, 0:1]
        vis  = x[:, 1:1+DIM_VIS]
        spec = x[:, 1+DIM_VIS:1+DIM_VIS+DIM_SPEC]
        imu  = x[:, 1+DIM_VIS+DIM_SPEC:1+DIM_VIS+DIM_SPEC+DIM_IMU]
        temp = x[:, -DIM_TEMP:]

        e_vis  = self.enc_vis(vis)
        e_spec = self.enc_spec(spec)
        e_imu  = self.enc_imu(imu)
        e_temp = self.enc_temp(temp)

        fused, w = self.fusion([e_vis, e_spec, e_imu, e_temp])
        out = self.decoder(fused)
        return out, w

model = SensorFusionV2().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[Modelo] SensorFusion v2  params={n_params:,}")
print(f"  MultiheadAttention({EMBED}, heads={NHEADS})")

# ── ENTRENAMIENTO ─────────────────────────────────────────────────────────────
EPOCHS = 8000; LR = 5e-4; BATCH = 256
LAM_DATA = 1.0; LAM_ODE = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

print(f"\n[Train] epochs={EPOCHS}  batch={BATCH}  lr={LR}")
print(f"  lam_data={LAM_DATA}  lam_ode={LAM_ODE}")
print(f"  {'Epoch':>6} {'L_data':>10} {'L_ode':>10} {'NRMSE_th':>10} {'w_vis':>8} {'w_spec':>8} {'w_imu':>8} {'w_tmp':>8}")
print("-"*84)

t_start = time.time()
best_nrmse = float('inf')
w_hist = []  # historial de pesos de atencion

for epoch in range(1, EPOCHS+1):
    # Batch aleatorio
    idx = torch.randperm(N, device=DEVICE)[:BATCH]
    x_b = X_t[idx]; y_b = Y_t[idx]
    t_b = t_t[idx]

    optimizer.zero_grad()
    pred, w = model(x_b)
    loss_data = torch.mean((pred - y_b)**2)

    # Residuo ODE: usar frames consecutivos ordenados temporalmente
    # Ordenar batch por tiempo para que las diferencias finitas sean validas
    idx_sort = torch.argsort(t_b)
    t_sorted  = t_b[idx_sort]
    pred_sorted, _ = model(x_b[idx_sort])

    # theta y omega predichos (desnormalizar)
    th_pred = pred_sorted[:,0] * Y_std[0] + Y_mean[0]
    om_pred = pred_sorted[:,1] * Y_std[1] + Y_mean[1]

    # Residuo ODE con diferencias finitas: dtheta/dt = omega
    dt_scaled = (t_sorted[1:] - t_sorted[:-1]) * t_end
    dt_scaled = torch.clamp(dt_scaled, min=1e-6)
    dth_dt = (th_pred[1:] - th_pred[:-1]) / dt_scaled
    res_ode = torch.mean((dth_dt - om_pred[:-1])**2)

    loss = LAM_DATA * loss_data + LAM_ODE * res_ode
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % (EPOCHS//10) == 0 or epoch == 1:
        with torch.no_grad():
            pred_all, w_all = model(X_t)
            th_pred_all = pred_all[:,0].cpu().numpy() * Y_std[0] + Y_mean[0]
            om_pred_all = pred_all[:,1].cpu().numpy() * Y_std[1] + Y_mean[1]
            nrmse_th = np.linalg.norm(th_pred_all-theta_true)/(np.linalg.norm(theta_true)+1e-10)
            nrmse_om = np.linalg.norm(om_pred_all-omega_true)/(np.linalg.norm(omega_true)+1e-10)
            w_mean = w_all.mean(dim=0).cpu().numpy()  # (4,) pesos promedio
            w_hist.append(w_mean.tolist())

        print(f"  {epoch:>6} {loss_data.item():>10.6f} {res_ode.item():>10.6f} "
              f"{nrmse_th:>10.6f} "
              f"{w_mean[0]:>8.4f} {w_mean[1]:>8.4f} {w_mean[2]:>8.4f} {w_mean[3]:>8.4f}")
        if nrmse_th < best_nrmse: best_nrmse = nrmse_th

# ── EVALUACION FINAL ──────────────────────────────────────────────────────────
with torch.no_grad():
    pred_all, w_all = model(X_t)
    th_pred = pred_all[:,0].cpu().numpy() * Y_std[0] + Y_mean[0]
    om_pred = pred_all[:,1].cpu().numpy() * Y_std[1] + Y_mean[1]
    w_final = w_all.mean(dim=0).cpu().numpy()

nrmse_th = np.linalg.norm(th_pred-theta_true)/(np.linalg.norm(theta_true)+1e-10)
nrmse_om = np.linalg.norm(om_pred-omega_true)/(np.linalg.norm(omega_true)+1e-10)

# Residuo ODE (diferencias finitas sobre toda la secuencia)
dt_arr  = np.diff(t_eval)
dth_num = np.diff(th_pred) / dt_arr
res_fd  = np.sqrt(np.mean((dth_num - om_pred[:-1])**2))

# Varianza de pesos de atencion (deben variar, no estar congelados)
w_array   = np.array(w_hist)  # (n_checkpoints, 4)
w_std_por_mod = w_array.std(axis=0)
pesos_activos = np.all(w_std_por_mod > 0.001)  # std > 0.1% = pesos cambian

nivel_th = "ALTA" if nrmse_th<0.01 else "MEDIA" if nrmse_th<0.05 else "FALLA"
nivel_om = "ALTA" if nrmse_om<0.01 else "MEDIA" if nrmse_om<0.05 else "FALLA"
nivel_fd = "ALTA" if res_fd<0.05  else "MEDIA" if res_fd<0.12  else "FALLA"

tf = time.time()-t_start
print(f"\n{'='*65}")
print(f"  RESULTADO — SensorFusion v2 (MultiheadAttention)")
print(f"{'='*65}")
print(f"  NRMSE theta : {nrmse_th:.6f}  [{nivel_th}]")
print(f"  NRMSE omega : {nrmse_om:.6f}  [{nivel_om}]")
print(f"  Residuo ODE : {res_fd:.6f}  [{nivel_fd}]")
print(f"  Pesos activos: {'SI' if pesos_activos else 'NO (congelados)'}")
print(f"  Pesos finales (vis|spec|imu|temp):")
print(f"    {w_final[0]:.4f} | {w_final[1]:.4f} | {w_final[2]:.4f} | {w_final[3]:.4f}")
print(f"  Varianza pesos durante entrenamiento:")
print(f"    vis={w_std_por_mod[0]:.4f} spec={w_std_por_mod[1]:.4f} "
      f"imu={w_std_por_mod[2]:.4f} temp={w_std_por_mod[3]:.4f}")
print(f"  Tiempo: {tf:.1f}s ({tf/60:.1f}min)")
print(f"  Device: {DEVICE}")
print(f"{'='*65}")

# Verificacion criterio Fase 5
ok = nivel_th in ("ALTA","MEDIA") and nivel_om in ("ALTA","MEDIA") and pesos_activos
print(f"\n  Criterio Fase 5.4: {'PASA' if ok else 'FALLA'}")
if not ok:
    if not pesos_activos:
        print("  [DEBUG] Pesos congelados — gradiente no llega a MHA")
    if nivel_th == "FALLA":
        print(f"  [DEBUG] NRMSE theta={nrmse_th:.4f} > 0.05")

# Guardar log
os.makedirs("runs_fase5", exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
lp = f"runs_fase5/sensorfusion_v2_{ts}.txt"
with open(lp, 'w') as f:
    f.write(f"SensorFusion v2|NRMSE_th={nrmse_th:.6f}|NRMSE_om={nrmse_om:.6f}\n")
    f.write(f"res_fd={res_fd:.6f}|pesos_activos={pesos_activos}\n")
    f.write(f"nivel_th={nivel_th}|nivel_om={nivel_om}|nivel_fd={nivel_fd}\n")
    f.write(f"w_final={w_final.tolist()}\n")
    f.write(f"tiempo={tf:.1f}s|device={DEVICE}\n")
print(f"  Log: {lp}")
