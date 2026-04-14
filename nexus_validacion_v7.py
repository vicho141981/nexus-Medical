# Copyright 2026 NEXUS Medical Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# nexus_validacion_v7.py
# NEXUS Medical - Validacion final vs Zhang et al. 2024
# IC cuadrada 10x10 (igual que el paper) + grid search analitico + PINN
# 20,000 epocas, guarda resultado por caso
import os, sys, json, math, time, datetime
import numpy as np

try:
    import torch, torch.nn as nn
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("[ERROR] pip install torch"); sys.exit(1)

print("="*65)
print("  NEXUS Medical - Validacion v7 Final")
print("  IC cuadrada + Grid Search analitico + PINN 20k")
print("  Ref: Zhang et al. arXiv:2311.16536")
print("="*65)
print(f"[Config] Device={DEVICE}")

NX = NY = 64; dx = 1.0/(NX-1)

CASOS = {
    "S1":{"D":0.003,"rho":0.150,"T":0.10},
    "S2":{"D":0.015,"rho":0.080,"T":0.30},
    "S3":{"D":0.005,"rho":0.300,"T":0.60},
    "S4":{"D":0.020,"rho":0.200,"T":0.30},
    "S5":{"D":0.008,"rho":0.050,"T":0.10},
    "S6":{"D":0.012,"rho":0.400,"T":0.30},
    "S7":{"D":0.002,"rho":0.100,"T":0.10},
    "S8":{"D":0.025,"rho":0.350,"T":0.20},
}

def simular(u0, D, rho, T, NT=200):
    u=u0.copy(); dt=T/NT
    for _ in range(NT):
        up=np.pad(u,1,mode='edge')
        lap=((up[2:,1:-1]-2*up[1:-1,1:-1]+up[:-2,1:-1])/dx**2+
             (up[1:-1,2:]-2*up[1:-1,1:-1]+up[1:-1,:-2])/dx**2)
        u=np.clip(u+dt*(D*lap+rho*u*(1-u)),0,1)
    return u.astype(np.float32)

def radio(u, th):
    m=(u>th).astype(float)
    return np.sqrt(m.sum()/np.pi) if m.sum()>4 else 0.0

def ic_cuadrada():
    u=np.zeros((NX,NY),dtype=np.float32)
    u[NX//2-5:NX//2+5, NY//2-5:NY//2+5]=1.0
    return u

def grid_search(u0, u1, T):
    """
    Con IC cuadrada, ningun observable geometrico da una estimacion
    confiable de D y rho por separado. D*rho desde v_frente tiene
    error aceptable (~20%) pero los clamps biologicos lo destruyen
    al mapear a D_init y rho_init.

    Decision: inicializacion fija neutral en el centro del espacio
    biologico. El PINN con 20k epocas desde ahi tiene margen suficiente.
    v_est se retorna para informacion pero no se usa para el init.
    """
    r05_t0 = radio(u0, 0.05)
    r05_t1 = radio(u1, 0.05)
    dr05   = max(r05_t1 - r05_t0, 0.5)
    v_est  = max((dr05 / NX) / T, 1e-4)

    # Punto neutro: centro geometrico del espacio log(D)*log(rho)
    # D en [0.0001,0.5] => centro log ~ 0.007
    # rho en [0.005, 5.0] => centro log ~ 0.16
    D_init   = 0.005
    rho_init = 0.10
    return D_init, rho_init, v_est, D_init * rho_init

class FE2D(nn.Module):
    def __init__(self):
        super().__init__()
        # [FIX] Generador local — no resetea estado global de PyTorch entre casos
        _g = torch.Generator(device='cpu')
        _g.manual_seed(42)
        self.register_buffer('Bx',torch.randn(1,32,generator=_g)*3.0)
        self.register_buffer('By',torch.randn(1,32,generator=_g)*3.0)
        self.register_buffer('Bt',torch.randn(1,16,generator=_g)*1.0)
    def forward(self,xyt):
        x=xyt[:,0:1];y=xyt[:,1:2];t=xyt[:,2:3]
        fx=torch.cat([torch.sin(2*math.pi*x@self.Bx),torch.cos(2*math.pi*x@self.Bx)],dim=1)
        fy=torch.cat([torch.sin(2*math.pi*y@self.By),torch.cos(2*math.pi*y@self.By)],dim=1)
        ft=torch.cat([torch.sin(2*math.pi*t@self.Bt),torch.cos(2*math.pi*t@self.Bt)],dim=1)
        return torch.cat([fx,fy,ft],dim=1)

class TNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=FE2D()
        self.net=nn.Sequential(
            nn.Linear(160,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),
            nn.Linear(256,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.sig=nn.Sigmoid()
    def forward(self,xyt): return self.sig(self.net(self.emb(xyt)))

def run_pinn(u0, umid, u1, D_init, rho_init, D_GT, rho_GT, T, epochs=20000):
    x1d=np.linspace(0,1,NX); y1d=np.linspace(0,1,NY)
    XX,YY=np.meshgrid(x1d,y1d,indexing='ij')

    def to_t(u,t):
        pts=np.stack([XX.flatten(),YY.flatten(),np.full(NX*NY,t),u.flatten()],axis=1)
        return (torch.tensor(pts[:,:3],dtype=torch.float32).to(DEVICE),
                torch.tensor(pts[:,3:4],dtype=torch.float32).to(DEVICE))

    X0,U0  = to_t(u0,   0.0)
    Xm,Um  = to_t(umid, 0.5)
    X1,U1  = to_t(u1,   1.0)
    X_ic   = torch.tensor(np.stack([XX.flatten(),YY.flatten(),np.zeros(NX*NY)],axis=1),
                          dtype=torch.float32).to(DEVICE)
    U_ic   = torch.tensor(u0.flatten().reshape(-1,1),dtype=torch.float32).to(DEVICE)

    D_MIN=0.0001; D_MAX=0.5; R_MIN=0.005; R_MAX=5.0

    log_D   = nn.Parameter(torch.tensor([math.log(max(D_init,D_MIN))],
                           dtype=torch.float32).to(DEVICE))
    log_rho = nn.Parameter(torch.tensor([math.log(max(rho_init,R_MIN))],
                           dtype=torch.float32).to(DEVICE))

    def get_p():
        return (torch.exp(torch.clamp(log_D,  math.log(D_MIN),math.log(D_MAX))),
                torch.exp(torch.clamp(log_rho,math.log(R_MIN),math.log(R_MAX))))

    model=TNet().to(DEVICE)
    opt_n=torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_p=torch.optim.Adam([log_D,log_rho],    lr=5e-3)
    sch_n=torch.optim.lr_scheduler.CosineAnnealingLR(opt_n,T_max=epochs,eta_min=1e-5)
    sch_p=torch.optim.lr_scheduler.CosineAnnealingLR(opt_p,T_max=epochs,eta_min=1e-5)

    def pde_res(xyt,Dp,rp):
        xyt=xyt.clone().requires_grad_(True); u=model(xyt)
        g=torch.autograd.grad(u,xyt,torch.ones_like(u),create_graph=True)[0]
        u_t=g[:,2:3];u_x=g[:,0:1];u_y=g[:,1:2]
        u_xx=torch.autograd.grad(u_x,xyt,torch.ones_like(u_x),create_graph=True)[0][:,0:1]
        u_yy=torch.autograd.grad(u_y,xyt,torch.ones_like(u_y),create_graph=True)[0][:,1:2]
        return u_t - T*(Dp*(u_xx+u_yy)+rp*u*(1-u))

    BATCH=1024; STAGE1=2000; PROG_END=6000; pde_f=1.0
    D_hist=[]; rho_hist=[]

    for epoch in range(1,epochs+1):
        opt_n.zero_grad(); opt_p.zero_grad()
        Dp,rp=get_p()
        loss_ic  = torch.mean((model(X_ic)-U_ic)**2)
        loss_img = (torch.mean((model(X0)-U0)**2)+
                    torch.mean((model(Xm)-Um)**2)+
                    torch.mean((model(X1)-U1)**2))

        if epoch>STAGE1:
            pts=torch.rand(BATCH,3,device=DEVICE)
            res=pde_res(pts,Dp,rp)
            loss_pde=torch.mean(res**2)
            prog=min(1.0,(epoch-STAGE1)/float(PROG_END))
            pde_f=float(loss_pde.detach())
            loss=80.0*loss_ic+80.0*loss_img+50.0*prog*loss_pde
        else:
            loss_pde=torch.tensor(0.0,device=DEVICE); prog=0.0
            loss=80.0*loss_ic+80.0*loss_img

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt_n.step(); sch_n.step()
        if epoch>STAGE1:
            opt_p.step(); sch_p.step()

        if epoch>(epochs-3000):
            with torch.no_grad():
                De,re=get_p()
                D_hist.append(float(De.detach()))
                rho_hist.append(float(re.detach()))

        if epoch%(epochs//10)==0:
            with torch.no_grad():
                De,re=get_p()
                De=float(De.detach()); re=float(re.detach())
            print(f"    Ep{epoch:6d} | D={De:.5f}({De/D_GT:.2f}x) "
                  f"rho={re:.5f}({re/rho_GT:.2f}x) PDE={pde_f:.2e}")

    with torch.no_grad():
        Df,rf=get_p()
        pred0=model(X0).cpu().numpy().flatten()
        nrmse=float(np.linalg.norm(pred0-u0.flatten())/
                    (np.linalg.norm(u0.flatten())+1e-10))

    conv=(len(D_hist)>0 and
          np.std(D_hist)/(np.mean(D_hist)+1e-10)<0.10 and
          np.std(rho_hist)/(np.mean(rho_hist)+1e-10)<0.10)

    return float(Df.detach()), float(rf.detach()), pde_f, nrmse, conv

# Correr 8 casos
resultados=[]; t0_total=time.time()
os.makedirs("runs_medico", exist_ok=True)

print(f"\n[Corriendo] 8 casos x 20000 epocas — IC cuadrada + Grid Search + PINN")

for nombre,g in CASOS.items():
    D_GT=g['D']; rho_GT=g['rho']; T=g['T']
    print(f"\n{'='*65}")
    print(f"  {nombre}: D_GT={D_GT:.4f}  rho_GT={rho_GT:.4f}  T={T}")
    t0c=time.time()

    # IC cuadrada igual que el paper
    u0   = ic_cuadrada()
    umid = simular(u0, D_GT, rho_GT, T/2)
    u1   = simular(umid, D_GT, rho_GT, T/2)

    cob0=np.mean(u0>0.1)*100; cob1=np.mean(u1>0.1)*100
    print(f"  Cob t0={cob0:.1f}% -> t1={cob1:.1f}%  delta={np.abs(u1-u0).mean():.5f}")

    # Grid search analitico
    D_init, rho_init, v_est, div_est = grid_search(u0, u1, T)
    err_D_init  = abs(D_init-D_GT)/D_GT*100
    err_rho_init= abs(rho_init-rho_GT)/rho_GT*100
    print(f"  Grid: D_init={D_init:.5f}({err_D_init:.0f}%) "
          f"rho_init={rho_init:.5f}({err_rho_init:.0f}%)")

    # PINN fine-tuning
    D_inf,rho_inf,pde_f,nrmse,conv = run_pinn(
        u0,umid,u1,D_init,rho_init,D_GT,rho_GT,T,epochs=20000)

    err_D  =abs(D_inf-D_GT)/D_GT*100
    err_rho=abs(rho_inf-rho_GT)/rho_GT*100
    div_GT =D_GT/rho_GT; div_inf=D_inf/rho_inf
    err_div=abs(div_inf-div_GT)/div_GT*100
    v_GT   =2*math.sqrt(D_GT*rho_GT); v_inf=2*math.sqrt(D_inf*rho_inf)
    err_v  =abs(v_inf-v_GT)/v_GT*100
    tf=time.time()-t0c

    nD  ="ALTA" if err_D<15   else "MEDIA" if err_D<30   else "BAJA"
    nR  ="ALTA" if err_rho<15 else "MEDIA" if err_rho<30 else "BAJA"
    nDiv="ALTA" if err_div<15 else "MEDIA" if err_div<30 else "BAJA"

    print(f"  D_inf  ={D_inf:.5f}  GT={D_GT:.5f}  err={err_D:.1f}%  [{nD}]")
    print(f"  rho_inf={rho_inf:.5f}  GT={rho_GT:.5f}  err={err_rho:.1f}%  [{nR}]")
    print(f"  D/rho  ={div_inf:.5f}  GT={div_GT:.5f}  err={err_div:.1f}%  [{nDiv}]")
    print(f"  PDE={pde_f:.2e}  NRMSE={nrmse:.4f}  Conv={'SI' if conv else 'NO'}  t={tf:.0f}s")

    r={'caso':nombre,'D_GT':D_GT,'rho_GT':rho_GT,'div_GT':div_GT,'v_GT':v_GT,
       'D_init':D_init,'rho_init':rho_init,
       'D_inf':D_inf,'rho_inf':rho_inf,'div_inf':div_inf,'v_inf':v_inf,
       'err_D':err_D,'err_rho':err_rho,'err_div':err_div,'err_v':err_v,
       'pde':pde_f,'nrmse':nrmse,'conv':conv,'tf':tf,'nD':nD,'nR':nR,'nDiv':nDiv}
    resultados.append(r)

    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"runs_medico/v7_{nombre}_{ts}.json",'w') as f: json.dump(r,f,indent=2)

# Resumen
tf_total=time.time()-t0_total
print(f"\n{'='*65}")
print(f"  RESUMEN FINAL v7")
print(f"{'='*65}")
print(f"  {'Caso':<6} {'errD%':>7} {'errR%':>7} {'errDiv%':>9} {'errV%':>7}  D_niv  rho_niv  Div_niv")
print(f"  {'-'*65}")
for r in resultados:
    print(f"  {r['caso']:<6} {r['err_D']:>7.1f} {r['err_rho']:>7.1f} "
          f"{r['err_div']:>9.1f} {r['err_v']:>7.1f}  {r['nD']:<6} {r['nR']:<8} {r['nDiv']}")

eD=[r['err_D'] for r in resultados]
eR=[r['err_rho'] for r in resultados]
eDiv=[r['err_div'] for r in resultados]
print(f"\n  Error medio D    : {np.mean(eD):.1f}% +/- {np.std(eD):.1f}%")
print(f"  Error medio rho  : {np.mean(eR):.1f}% +/- {np.std(eR):.1f}%")
print(f"  Error medio D/rho: {np.mean(eDiv):.1f}% +/- {np.std(eDiv):.1f}%")
n_alta_div=sum(1 for r in resultados if r['nDiv']=='ALTA')
print(f"\n  ALTA (<15%): D/rho={n_alta_div}/8")
print(f"  Tiempo total: {tf_total:.0f}s ({tf_total/60:.1f}min)  Device={DEVICE}")

print(f"\n  Paper Zhang et al. 2024: errores D/rho ~10-25% con SEG only")
print(f"  NEXUS v7: {np.mean(eDiv):.1f}%")
if np.mean(eDiv)<25:   print("  CONCLUSION: COMPARABLE al paper ✅")
elif np.mean(eDiv)<40: print("  CONCLUSION: CERCANO al paper")
else:                  print("  CONCLUSION: Mejorar grid search")

ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f"runs_medico/v7_FINAL_{ts}.json",'w') as f:
    json.dump({'version':'v7','epochs':20000,'device':str(DEVICE),
               'resultados':resultados,
               'err_D':float(np.mean(eD)),'err_rho':float(np.mean(eR)),
               'err_div':float(np.mean(eDiv))},f,indent=2)
print(f"  Log: runs_medico/v7_FINAL_{ts}.json")
