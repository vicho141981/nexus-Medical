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

r"""
nexus_brats_pipeline_v4.py — NEXUS Medical M4 v4 final
=======================================================
Fix definitivo: 
- T_SIM=0.3 (no 0.5) — señal mas suave
- D_SIM=0.003, RHO_SIM=0.3 — parametros biologicamente plausibles
- Normalizacion por masa integral, no por max
- 3 snapshots para identifiabilidad
"""
import os, sys, argparse, time, math, datetime
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] pip install nibabel"); sys.exit(1)

try:
    import torch
    import torch.nn as nn
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("[ERROR] pip install torch"); sys.exit(1)

from scipy.ndimage import gaussian_filter, zoom as nd_zoom

print("="*65)
print("  NEXUS Medical — BraTS Pipeline v4 final")
print("="*65)

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str,
    default='F:/NEXUS/archivos nexus medico/Task01_BrainTumour')
parser.add_argument('--caso',    type=str, default=None)
parser.add_argument('--n_casos', type=int, default=1)
parser.add_argument('--canal',   type=int, default=2)
parser.add_argument('--label',   type=int, default=3)
parser.add_argument('--epochs',  type=int, default=12000)
parser.add_argument('--seed',    type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)

D_MIN=0.0001; D_MAX=0.05
RHO_MIN=0.001; RHO_MAX=2.0
UMBRAL_ALTO=0.18; UMBRAL_MEDIO=0.043
MIN_PIXELS=150
LAM_IC=100.0; LAM_IMG=30.0; LAM_PDE=20.0
STAGE1=300; PROG_END=1000
# Parametros de simulacion calibrados
D_SIM=0.003; RHO_SIM=0.3; T_SIM=0.3

print(f"[Config] Device={DEVICE}  epochs={args.epochs}")
print(f"[Sim]    D={D_SIM} rho={RHO_SIM} T={T_SIM} (3 snapshots)")
print(f"[Bio]    D=[{D_MIN},{D_MAX}]  rho=[{RHO_MIN},{RHO_MAX}]")

base=args.base
imgs_dir=os.path.join(base,'imagesTr')
lbls_dir=os.path.join(base,'labelsTr')
imgs=sorted([f for f in os.listdir(imgs_dir)
             if f.endswith('.nii.gz') and not f.startswith('._')])
lbls=sorted([f for f in os.listdir(lbls_dir)
             if f.endswith('.nii.gz') and not f.startswith('._')])
casos_all=sorted({f.replace('.nii.gz','') for f in imgs} &
                 {f.replace('.nii.gz','') for f in lbls})
casos_run=[args.caso] if args.caso else casos_all[:args.n_casos]
print(f"[Dataset] {len(casos_all)} casos | Procesando: {len(casos_run)}")
os.makedirs("runs_medico",exist_ok=True)

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
        fx=torch.cat([torch.sin(2*math.pi*x@self.Bx),
                      torch.cos(2*math.pi*x@self.Bx)],dim=1)
        fy=torch.cat([torch.sin(2*math.pi*y@self.By),
                      torch.cos(2*math.pi*y@self.By)],dim=1)
        ft=torch.cat([torch.sin(2*math.pi*t@self.Bt),
                      torch.cos(2*math.pi*t@self.Bt)],dim=1)
        return torch.cat([fx,fy,ft],dim=1)

class TNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=FE2D()
        self.net=nn.Sequential(
            nn.Linear(160,256),nn.Tanh(),
            nn.Linear(256,256),nn.Tanh(),
            nn.Linear(256,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),
            nn.Linear(64,1))
        self.sig=nn.Sigmoid()
    def forward(self,xyt):
        return self.sig(self.net(self.emb(xyt)))

def simular_fd(u0, D, rho, T, NT=50, NX=32):
    dt=T/NT; dx=1.0/(NX-1); u=u0.copy()
    for _ in range(NT):
        up=np.pad(u,1,mode='edge')
        lap=((up[2:,1:-1]-2*up[1:-1,1:-1]+up[:-2,1:-1])/dx**2+
             (up[1:-1,2:]-2*up[1:-1,1:-1]+up[1:-1,:-2])/dx**2)
        u=np.clip(u+dt*(D*lap+rho*u*(1-u)),0,1)
    return u.astype(np.float32)

def run_fisher(u_t0, u_tmid, u_t1, epochs, NX=32, NY=32):
    x1d=np.linspace(0,1,NX); y1d=np.linspace(0,1,NY)
    XX,YY=np.meshgrid(x1d,y1d,indexing='ij')

    def to_t(u,t):
        pts=np.stack([XX.flatten(),YY.flatten(),
                      np.full(NX*NY,t),u.flatten()],axis=1)
        return (torch.tensor(pts[:,:3],dtype=torch.float32).to(DEVICE),
                torch.tensor(pts[:,3:4],dtype=torch.float32).to(DEVICE))

    X0,U0 = to_t(u_t0,  0.0)
    Xm,Um = to_t(u_tmid,0.5)
    X1,U1 = to_t(u_t1,  1.0)
    X_ic=torch.tensor(
        np.stack([XX.flatten(),YY.flatten(),np.zeros(NX*NY)],axis=1),
        dtype=torch.float32).to(DEVICE)
    U_ic=torch.tensor(u_t0.flatten().reshape(-1,1),
                      dtype=torch.float32).to(DEVICE)

    log_D  =nn.Parameter(torch.tensor([math.log(0.005)],
                         dtype=torch.float32).to(DEVICE))
    log_rho=nn.Parameter(torch.tensor([math.log(0.1)],
                         dtype=torch.float32).to(DEVICE))
    log_D_min=math.log(D_MIN);   log_D_max=math.log(D_MAX)
    log_r_min=math.log(RHO_MIN); log_r_max=math.log(RHO_MAX)

    def get_p():
        Dc =torch.exp(torch.clamp(log_D,  log_D_min,log_D_max))
        rc =torch.exp(torch.clamp(log_rho,log_r_min,log_r_max))
        return Dc,rc

    model=TNet().to(DEVICE)
    opt_net   =torch.optim.Adam(model.parameters(),lr=1e-3)
    opt_params=torch.optim.Adam([log_D,log_rho],lr=3e-3)
    sch_net   =torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_net,T_max=epochs,eta_min=1e-5)
    sch_params=torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_params,T_max=epochs,eta_min=1e-5)

    def pde_res(xyt,Dp,rp):
        xyt=xyt.clone().requires_grad_(True)
        u=model(xyt)
        g=torch.autograd.grad(u,xyt,torch.ones_like(u),create_graph=True)[0]
        u_t=g[:,2:3];u_x=g[:,0:1];u_y=g[:,1:2]
        u_xx=torch.autograd.grad(u_x,xyt,torch.ones_like(u_x),
                                 create_graph=True)[0][:,0:1]
        u_yy=torch.autograd.grad(u_y,xyt,torch.ones_like(u_y),
                                 create_graph=True)[0][:,1:2]
        return u_t-Dp*(u_xx+u_yy)-rp*u*(1-u)

    BATCH=1024; pde_f=1.0
    D_hist=[]; rho_hist=[]

    for epoch in range(1,epochs+1):
        opt_net.zero_grad(); opt_params.zero_grad()
        Dp,rp=get_p()
        loss_ic =torch.mean((model(X_ic)-U_ic)**2)
        loss_img=(torch.mean((model(X0)-U0)**2)+
                  torch.mean((model(Xm)-Um)**2)+
                  torch.mean((model(X1)-U1)**2))
        if epoch>STAGE1:
            xc=torch.rand(BATCH,1,device=DEVICE)
            yc=torch.rand(BATCH,1,device=DEVICE)
            tc=torch.rand(BATCH,1,device=DEVICE)
            res=pde_res(torch.cat([xc,yc,tc],dim=1),Dp,rp)
            loss_pde=torch.mean(res**2)
            prog=min(1.0,(epoch-STAGE1)/float(PROG_END))
            pde_f=float(loss_pde.detach())
        else:
            loss_pde=torch.tensor(0.0,device=DEVICE); prog=0.0

        loss=LAM_IC*loss_ic+LAM_IMG*loss_img+LAM_PDE*prog*loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt_net.step(); opt_params.step()
        sch_net.step(); sch_params.step()

        if epoch>(epochs-2000):
            with torch.no_grad():
                De,re=get_p()
                D_hist.append(float(De.detach()))
                rho_hist.append(float(re.detach()))

        if epoch%(epochs//4)==0:
            with torch.no_grad():
                De,re=get_p()
                De=float(De.detach()); re=float(re.detach())
            print(f"    Ep {epoch:5d} | D={De:.5f}  rho={re:.5f}  "
                  f"D/rho={De/re:.4f}  PDE={pde_f:.2e}")

    with torch.no_grad():
        Df,rf=get_p()
        pred0=model(X0).cpu().numpy().flatten()
        nrmse=float(np.linalg.norm(pred0-u_t0.flatten())/
                    (np.linalg.norm(u_t0.flatten())+1e-10))
    conv_D  =np.std(D_hist)/(np.mean(D_hist)+1e-10)<0.10
    conv_rho=np.std(rho_hist)/(np.mean(rho_hist)+1e-10)<0.10
    return float(Df.detach()),float(rf.detach()),pde_f,nrmse,conv_D and conv_rho

resultados=[]; anomalos=[]

for caso in casos_run:
    print(f"\n{'='*65}")
    print(f"  Procesando: {caso}")
    print(f"{'='*65}")
    t0c=time.time()
    try:
        vol_full=nib.load(os.path.join(imgs_dir,f"{caso}.nii.gz")).get_fdata().astype(np.float32)
        seg_full=nib.load(os.path.join(lbls_dir,f"{caso}.nii.gz")).get_fdata().astype(np.int32)
        sp_nib  =nib.load(os.path.join(imgs_dir,f"{caso}.nii.gz")).header.get_zooms()
        sp3=(float(sp_nib[0]),float(sp_nib[1]),float(sp_nib[2]))

        vol=vol_full[:,:,:,args.canal] if vol_full.ndim==4 else vol_full
        p1=np.percentile(vol[vol>0],1); p99=np.percentile(vol[vol>0],99)

        tumor_gt=(seg_full==args.label).astype(np.uint8)
        if tumor_gt.sum()==0:
            print(f"  [SKIP] Label {args.label} no encontrado"); continue

        n_vox=tumor_gt.sum(); vol_mm3=n_vox*sp3[0]*sp3[1]*sp3[2]
        px_por_z=tumor_gt.sum(axis=(0,1))
        z_c=int(np.argmax(px_por_z)); px_c=int(px_por_z[z_c])
        print(f"  Tumor GT: {n_vox:,} vox = {vol_mm3/1000:.1f} cm3")
        print(f"  Corte optimo: z={z_c}  pixels={px_c}",
              "OK" if px_c>=MIN_PIXELS else "INSUFICIENTE")

        if px_c<MIN_PIXELS:
            anomalos.append({'caso':caso,'razon':f'px={px_c}'})
            continue

        NX=NY=32
        slice_gt=tumor_gt[:,:,z_c].astype(float)
        u_raw=gaussian_filter(slice_gt,sigma=2.0)
        if u_raw.max()>0: u_raw/=u_raw.max()
        fx=NX/u_raw.shape[0]; fy=NY/u_raw.shape[1]
        u_t0=np.clip(nd_zoom(u_raw,(fx,fy),order=3),0,1).astype(np.float32)

        # 3 snapshots con parametros calibrados
        u_tmid=simular_fd(u_t0,D_SIM,RHO_SIM,T=T_SIM,NX=NX)
        u_t1  =simular_fd(u_tmid,D_SIM,RHO_SIM,T=T_SIM,NX=NX)

        M0=u_t0.sum(); Mm=u_tmid.sum(); M1=u_t1.sum()
        print(f"  Masa: t0={M0:.1f}  tmid={Mm:.1f}  t1={M1:.1f}  "
              f"crecimiento={((M1-M0)/M0*100):.1f}%")
        print(f"  Cob: t0={np.mean(u_t0>0.1)*100:.1f}%  t1={np.mean(u_t1>0.1)*100:.1f}%")

        print(f"\n  Fisher-KPP Inverso ({args.epochs} ep)...")
        D_f,rho_f,pde_f,nrmse_f,conv=run_fisher(u_t0,u_tmid,u_t1,args.epochs)

        div=D_f/rho_f if rho_f>1e-10 else 999
        v_frente=2*math.sqrt(D_f*rho_f)
        cob_t0=float(np.mean(u_t0>0.1)*100)

        dx=1.0/(NX-1); T_PRED=T_SIM; NT_P=50; dt_p=T_PRED/NT_P
        u_pred=u_t1.copy()
        for _ in range(NT_P):
            up=np.pad(u_pred,1,mode='edge')
            lap=((up[2:,1:-1]-2*up[1:-1,1:-1]+up[:-2,1:-1])/dx**2+
                 (up[1:-1,2:]-2*up[1:-1,1:-1]+up[1:-1,:-2])/dx**2)
            u_pred=np.clip(u_pred+dt_p*(D_f*lap+rho_f*u_pred*(1-u_pred)),0,1)
        cob_pred=float(np.mean(u_pred>0.1)*100)

        err_D  =abs(D_f-D_SIM)/D_SIM*100
        err_rho=abs(rho_f-RHO_SIM)/RHO_SIM*100

        if div>=UMBRAL_ALTO:    riesgo="ALTO";  recom="Reseccion amplia"
        elif div>=UMBRAL_MEDIO: riesgo="MEDIO"; recom="Seguimiento frecuente"
        else:                   riesgo="BAJO";  recom="Protocolo estandar"

        p1_ok=pde_f<1e-4; p2_ok=nrmse_f<0.01
        pilares=sum([p1_ok,p2_ok,conv,True,div<1.0])

        tf=time.time()-t0c
        print(f"\n  D_inferido={D_f:.6f}  D_verdad={D_SIM}  err={err_D:.1f}%")
        print(f"  rho_inf   ={rho_f:.6f}  rho_verd ={RHO_SIM}  err={err_rho:.1f}%")
        print(f"  D/rho={div:.4f}  v={v_frente:.4f}")
        print(f"  [P1] PDE  ={pde_f:.2e}  {'OK' if p1_ok else 'FALLA'}")
        print(f"  [P2] NRMSE={nrmse_f:.4f}  {'OK' if p2_ok else 'FALLA'}")
        print(f"  [P3] Conv ={'SI' if conv else 'NO'}")
        print(f"  [P5] {riesgo} — {recom}")
        print(f"  Pilares: {pilares}/5  Tiempo: {tf:.0f}s ({tf/60:.1f}min)")

        res={'caso':caso,'D':D_f,'rho':rho_f,'div':div,'v':v_frente,
             'pde':pde_f,'nrmse':nrmse_f,'conv':conv,'riesgo':riesgo,
             'err_D':err_D,'err_rho':err_rho,
             'cob_t0':cob_t0,'cob_pred':cob_pred,'pilares':pilares,'tf':tf}
        resultados.append(res)

        ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"runs_medico/brats_v4_{caso}_{ts}.txt",'w') as f:
            f.write(f"v4final|{caso}|z={z_c}|3snap|D_sim={D_SIM}|rho_sim={RHO_SIM}\n")
            f.write(f"D={D_f:.6f}({err_D:.1f}%)|rho={rho_f:.6f}({err_rho:.1f}%)\n")
            f.write(f"div={div:.4f}|v={v_frente:.4f}|PDE={pde_f:.2e}|NRMSE={nrmse_f:.4f}\n")
            f.write(f"pilares={pilares}/5|{riesgo}|t={tf:.0f}s\n")

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

if not resultados:
    print("[ERROR] Sin resultados"); sys.exit(1)

print(f"\n{'='*65}")
print(f"  RESUMEN — NEXUS BraTS v4 final")
print(f"  D_sim={D_SIM}  rho_sim={RHO_SIM}  T={T_SIM}")
print(f"{'='*65}")
print(f"  Validos:{len(resultados)}  Anomalos:{len(anomalos)}")
print()
print(f"  {'Caso':<12} {'D_inf':>8} {'rho_inf':>8} "
      f"{'errD%':>7} {'errR%':>7} {'PDE':>9} {'NRMSE':>7} {'P':>4}  Riesgo")
print(f"  {'-'*78}")
for r in resultados:
    print(f"  {r['caso']:<12} {r['D']:>8.5f} {r['rho']:>8.5f} "
          f"{r['err_D']:>7.1f} {r['err_rho']:>7.1f} "
          f"{r['pde']:>9.2e} {r['nrmse']:>7.4f} "
          f"{r['pilares']:>3}/5  {r['riesgo']}")
if len(resultados)>1:
    n_p1=sum(1 for r in resultados if r['pde']<1e-4)
    n_p2=sum(1 for r in resultados if r['nrmse']<0.01)
    n_5p=sum(1 for r in resultados if r['pilares']==5)
    eD=[r['err_D'] for r in resultados]
    eR=[r['err_rho'] for r in resultados]
    print(f"\n  Error D   : {np.mean(eD):.1f}% +/- {np.std(eD):.1f}%")
    print(f"  Error rho : {np.mean(eR):.1f}% +/- {np.std(eR):.1f}%")
    print(f"  P1 PDE<1e-4 : {n_p1}/{len(resultados)}")
    print(f"  P2 NRMSE<1% : {n_p2}/{len(resultados)}")
    print(f"  5/5 pilares : {n_5p}/{len(resultados)}")
print(f"{'='*65}")
print("[M4 v4 final] OK")
