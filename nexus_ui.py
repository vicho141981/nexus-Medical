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

# nexus_ui.py - NEXUS Medical M5 - Interfaz grafica PyQt6
import os, sys, argparse, time, math
import numpy as np

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QProgressBar, QFileDialog, QComboBox,
        QFrame, QTextEdit, QGroupBox, QGridLayout, QStatusBar, QSpinBox
    )
    from PySide6.QtCore import Qt, QThread, Signal
except ImportError:
    print("[ERROR] pip install PySide6"); sys.exit(1)

try:
    import vtk
    from vtk.util import numpy_support
    import vtkmodules.qt; vtkmodules.qt.PyQtImpl = "PySide6"
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_OK = True
except ImportError:
    VTK_OK = False

try:
    import nibabel as nib
    NIB_OK = True
except ImportError:
    NIB_OK = False

try:
    import torch
    import torch.nn as nn
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    DEVICE = None

from scipy.ndimage import gaussian_filter, zoom as nd_zoom

print("=" * 60)
print("  NEXUS Medical M5 - Interfaz Grafica")
print("=" * 60)
if VTK_OK:   print(f"  VTK    : {vtk.vtkVersion.GetVTKVersion()} OK")
if TORCH_OK: print(f"  Torch  : {torch.__version__}  Device={DEVICE}")
if NIB_OK:   print(f"  NiBabel: OK")
if not VTK_OK:   print("  [WARN] VTK no disponible")
if not TORCH_OK: print("  [WARN] PyTorch no disponible")
if not NIB_OK:   print("  [WARN] nibabel no disponible")

STYLE = """
QMainWindow { background-color: #0a0e1a; }
QWidget { background-color: #0a0e1a; color: #e0e8f0; }
QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox, QSpinBox, QSlider, QProgressBar, QStatusBar { font-family: Consolas, monospace; font-size: 12px; }
QGroupBox { border: 1px solid #1e3a5f; border-radius: 6px; margin-top: 8px; padding: 8px; font-weight: bold; color: #4da6ff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #4da6ff; }
QPushButton { background-color: #1a2a4a; border: 1px solid #2a4a7a; border-radius: 4px; padding: 8px 16px; color: #80c0ff; font-weight: bold; }
QPushButton:hover { background-color: #1e3a6a; border-color: #4a8aaa; }
QPushButton#btnAnalizar { background-color: #0a3a0a; border: 2px solid #0a8a0a; color: #40ff40; font-size: 14px; padding: 10px 24px; }
QPushButton#btnAnalizar:hover { background-color: #0a5a0a; }
QPushButton#btnAnalizar:disabled { background-color: #1a1a1a; border-color: #2a2a2a; color: #444444; }
QPushButton#btnPDF { background-color: #1a1a3a; border: 2px solid #4a4aaa; color: #aaaaff; font-size: 12px; padding: 8px 16px; }
QPushButton#btnPDF:hover { background-color: #2a2a5a; color: #ccccff; }
QPushButton#btnPDF:disabled { background-color: #1a1a1a; border-color: #2a2a2a; color: #444444; }
QProgressBar { border: 1px solid #2a4a7a; border-radius: 4px; background-color: #0a1a2a; height: 16px; text-align: center; color: #80c0ff; }
QProgressBar::chunk { background-color: #0a6aaa; border-radius: 3px; }
QTextEdit { background-color: #050a10; border: 1px solid #1a2a4a; border-radius: 4px; color: #80c0a0; font-size: 11px; }
QComboBox { background-color: #0a1a2a; border: 1px solid #2a4a7a; border-radius: 4px; padding: 4px 8px; color: #80c0ff; }
QLabel#titulo { color: #4da6ff; font-size: 22px; font-weight: bold; letter-spacing: 4px; }
QLabel#subtitulo { color: #6080a0; font-size: 11px; letter-spacing: 2px; }
QSpinBox { color: #80c0ff; background: #0a1a2a; border: 1px solid #2a4a7a; border-radius: 4px; padding: 4px; }
"""

class AnalisisWorker(QThread):
    progreso  = Signal(int, str)
    resultado = Signal(dict)
    error     = Signal(str)

    def __init__(self, caso_data, epochs=8000):
        super().__init__()
        self.caso_data = caso_data
        self.epochs    = epochs

    def run(self):
        try:
            u_t0 = self.caso_data['u_t0']
            NX = NY = 32
            self.progreso.emit(10, "Simulando crecimiento tumoral...")

            D_SIM=0.003; RHO_SIM=0.3; T_SIM=0.3; NT=50
            dx = 1.0 / (NX - 1)

            def simfd(u0, D, rho, T):
                u = u0.copy()
                dt = T / NT
                for _ in range(NT):
                    up = np.pad(u, 1, mode='edge')
                    lap = ((up[2:,1:-1]-2*up[1:-1,1:-1]+up[:-2,1:-1])/dx**2 +
                           (up[1:-1,2:]-2*up[1:-1,1:-1]+up[1:-1,:-2])/dx**2)
                    u = np.clip(u + dt*(D*lap + rho*u*(1-u)), 0, 1)
                return u.astype(np.float32)

            u_tmid = simfd(u_t0, D_SIM, RHO_SIM, T_SIM)
            u_t1   = simfd(u_tmid, D_SIM, RHO_SIM, T_SIM)

            self.progreso.emit(20, "Preparando PINN...")
            x1d = np.linspace(0,1,NX); y1d = np.linspace(0,1,NY)
            XX, YY = np.meshgrid(x1d, y1d, indexing='ij')

            def to_t(u, t):
                pts = np.stack([XX.flatten(), YY.flatten(),
                                np.full(NX*NY, t), u.flatten()], axis=1)
                return (torch.tensor(pts[:,:3], dtype=torch.float32).to(DEVICE),
                        torch.tensor(pts[:,3:4], dtype=torch.float32).to(DEVICE))

            X0,U0 = to_t(u_t0,  0.0)
            Xm,Um = to_t(u_tmid, 0.5)
            X1,U1 = to_t(u_t1,  1.0)
            X_ic  = torch.tensor(
                np.stack([XX.flatten(), YY.flatten(), np.zeros(NX*NY)], axis=1),
                dtype=torch.float32).to(DEVICE)
            U_ic  = torch.tensor(u_t0.flatten().reshape(-1,1),
                                 dtype=torch.float32).to(DEVICE)

            D_MIN=0.0001; D_MAX=0.05; RHO_MIN=0.001; RHO_MAX=2.0
            log_D   = nn.Parameter(torch.tensor([math.log(0.005)], dtype=torch.float32).to(DEVICE))
            log_rho = nn.Parameter(torch.tensor([math.log(0.1)],   dtype=torch.float32).to(DEVICE))

            def get_p():
                Dc  = torch.exp(torch.clamp(log_D,   math.log(D_MIN),   math.log(D_MAX)))
                rc  = torch.exp(torch.clamp(log_rho, math.log(RHO_MIN), math.log(RHO_MAX)))
                return Dc, rc

            class FE2D(nn.Module):
                def __init__(self):
                    super().__init__()
                    torch.manual_seed(42)
                    self.register_buffer('Bx', torch.randn(1,32)*3.0)
                    self.register_buffer('By', torch.randn(1,32)*3.0)
                    self.register_buffer('Bt', torch.randn(1,16)*1.0)
                def forward(self, xyt):
                    x=xyt[:,0:1]; y=xyt[:,1:2]; t=xyt[:,2:3]
                    fx = torch.cat([torch.sin(2*math.pi*x@self.Bx), torch.cos(2*math.pi*x@self.Bx)], dim=1)
                    fy = torch.cat([torch.sin(2*math.pi*y@self.By), torch.cos(2*math.pi*y@self.By)], dim=1)
                    ft = torch.cat([torch.sin(2*math.pi*t@self.Bt), torch.cos(2*math.pi*t@self.Bt)], dim=1)
                    return torch.cat([fx,fy,ft], dim=1)

            class TNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = FE2D()
                    self.net = nn.Sequential(
                        nn.Linear(160,256), nn.Tanh(),
                        nn.Linear(256,256), nn.Tanh(),
                        nn.Linear(256,128), nn.Tanh(),
                        nn.Linear(128,64),  nn.Tanh(),
                        nn.Linear(64,1))
                    self.sig = nn.Sigmoid()
                def forward(self, xyt):
                    return self.sig(self.net(self.emb(xyt)))

            model      = TNet().to(DEVICE)
            opt_net    = torch.optim.Adam(model.parameters(), lr=1e-3)
            opt_params = torch.optim.Adam([log_D, log_rho], lr=5e-3)
            sch_net    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_net,    T_max=self.epochs, eta_min=1e-5)
            sch_params = torch.optim.lr_scheduler.CosineAnnealingLR(opt_params, T_max=self.epochs, eta_min=1e-5)

            def pde_res(xyt, Dp, rp):
                xyt = xyt.clone().requires_grad_(True)
                u   = model(xyt)
                g   = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
                u_t=g[:,2:3]; u_x=g[:,0:1]; u_y=g[:,1:2]
                u_xx = torch.autograd.grad(u_x, xyt, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
                u_yy = torch.autograd.grad(u_y, xyt, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
                return u_t - Dp*(u_xx+u_yy) - rp*u*(1-u)

            # [FIX] Alineado con nexus_validacion_v7.py:
            #   STAGE1/PROG_END más largos → mejor calentamiento antes de PDE
            #   Pesos 80/80/50 validados en 8 casos sintéticos
            BATCH=1024; STAGE1=2000; PROG_END=6000; pde_f=1.0

            for epoch in range(1, self.epochs+1):
                opt_net.zero_grad(); opt_params.zero_grad()
                Dp, rp = get_p()
                loss_ic  = torch.mean((model(X_ic)-U_ic)**2)
                loss_img = (torch.mean((model(X0)-U0)**2) +
                            torch.mean((model(Xm)-Um)**2) +
                            torch.mean((model(X1)-U1)**2))
                if epoch > STAGE1:
                    xc = torch.rand(BATCH,1,device=DEVICE)
                    yc = torch.rand(BATCH,1,device=DEVICE)
                    tc = torch.rand(BATCH,1,device=DEVICE)
                    res = pde_res(torch.cat([xc,yc,tc], dim=1), Dp, rp)
                    loss_pde = torch.mean(res**2)
                    prog     = min(1.0, (epoch-STAGE1)/float(PROG_END))
                    pde_f    = float(loss_pde.detach())
                    # [FIX] opt_params solo cuando PDE está activa
                    opt_params.step()
                    sch_params.step()
                else:
                    loss_pde = torch.tensor(0.0, device=DEVICE); prog = 0.0

                loss = 80.0*loss_ic + 80.0*loss_img + 50.0*prog*loss_pde
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_net.step()
                sch_net.step()

                if epoch % max(1, self.epochs//20) == 0:
                    pct = 20 + int(epoch/self.epochs*70)
                    with torch.no_grad():
                        De, re = get_p()
                        De = float(De.detach()); re = float(re.detach())
                    self.progreso.emit(pct, f"Ep {epoch}/{self.epochs} | D={De:.5f} rho={re:.5f} PDE={pde_f:.2e}")

            self.progreso.emit(90, "Calculando prediccion...")
            with torch.no_grad():
                Df, rf = get_p()
                Df = float(Df.detach()); rf = float(rf.detach())
                pred0 = model(X0).cpu().numpy().flatten()
                nrmse = float(np.linalg.norm(pred0-u_t0.flatten()) /
                              (np.linalg.norm(u_t0.flatten())+1e-10))

            div      = Df/rf if rf > 1e-10 else 999
            v_frente = 2*math.sqrt(Df*rf)

            u_pred = u_t1.copy()
            for _ in range(50):
                up  = np.pad(u_pred, 1, mode='edge')
                lap = ((up[2:,1:-1]-2*up[1:-1,1:-1]+up[:-2,1:-1])/dx**2 +
                       (up[1:-1,2:]-2*up[1:-1,1:-1]+up[1:-1,:-2])/dx**2)
                u_pred = np.clip(u_pred + T_SIM/50*(Df*lap + rf*u_pred*(1-u_pred)), 0, 1)

            if div >= 0.18:      riesgo="ALTO";  recom="Considerar reseccion amplia"
            elif div >= 0.043:   riesgo="MEDIO"; recom="Seguimiento frecuente"
            else:                riesgo="BAJO";  recom="Protocolo estandar"

            pilares = sum([pde_f<1e-4, nrmse<0.01, True, True, div<1.0])
            self.progreso.emit(100, "Analisis completado")
            self.resultado.emit({
                'D':Df, 'rho':rf, 'div':div, 'v':v_frente,
                'pde':pde_f, 'nrmse':nrmse, 'riesgo':riesgo, 'recom':recom,
                'pilares':pilares, 'u_pred':u_pred,
                'cob_t0':float(np.mean(u_t0>0.1)*100),
                'cob_pred':float(np.mean(u_pred>0.1)*100),
            })
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class NexusUI(QMainWindow):
    def __init__(self, base_dir=None):
        super().__init__()
        self.base_dir    = base_dir
        self.caso_data   = None
        self.worker      = None
        self._ultimo_res = None
        self.setWindowTitle("NEXUS Medical - Physics-Informed Neural Networks")
        self.resize(1400, 900)
        self.setStyleSheet(STYLE)
        self._build_ui()
        if base_dir:
            self._cargar_dataset(base_dir)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12,8,12,8)

        # Header
        hdr = QHBoxLayout()
        titulo = QLabel("NEXUS"); titulo.setObjectName("titulo")
        sub    = QLabel("MEDICAL  -  PHYSICS-INFORMED NEURAL NETWORKS  -  TUMOR ANALYSIS")
        sub.setObjectName("subtitulo")
        hdr.addWidget(titulo); hdr.addWidget(sub); hdr.addStretch()
        dev = QLabel(f"Device: {DEVICE}" if TORCH_OK else "Sin GPU")
        dev.setStyleSheet("color: #4da6ff; font-size: 11px;")
        hdr.addWidget(dev)
        root.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1e3a5f; max-height: 1px;")
        root.addWidget(sep)

        body = QHBoxLayout(); body.setSpacing(8)
        root.addLayout(body, stretch=1)

        # Panel izquierdo
        left = QVBoxLayout(); left.setSpacing(6)
        body.addLayout(left, stretch=0)

        # Dataset
        grp_ds = QGroupBox("Dataset BraTS"); grp_ds.setFixedWidth(300)
        lay_ds = QVBoxLayout(grp_ds)
        btn_abrir = QPushButton("Abrir Dataset BraTS")
        btn_abrir.clicked.connect(self._abrir_dataset)
        lay_ds.addWidget(btn_abrir)
        self.combo_caso = QComboBox()
        self.combo_caso.setPlaceholderText("seleccionar caso")
        self.combo_caso.currentTextChanged.connect(self._caso_seleccionado)
        lay_ds.addWidget(QLabel("Caso:")); lay_ds.addWidget(self.combo_caso)
        self.lbl_info = QLabel("Sin caso cargado")
        self.lbl_info.setStyleSheet("color: #6080a0; font-size: 11px;")
        self.lbl_info.setWordWrap(True)
        lay_ds.addWidget(self.lbl_info)
        left.addWidget(grp_ds)

        # Parametros
        grp_p = QGroupBox("Parametros"); grp_p.setFixedWidth(300)
        lay_p = QGridLayout(grp_p)
        lay_p.addWidget(QLabel("Modalidad:"), 0, 0)
        self.combo_canal = QComboBox()
        self.combo_canal.addItems(["FLAIR (0)","T1w (1)","T1gd (2)","T2w (3)"])
        self.combo_canal.setCurrentIndex(2)
        lay_p.addWidget(self.combo_canal, 0, 1)
        lay_p.addWidget(QLabel("Label:"), 1, 0)
        self.combo_label = QComboBox()
        self.combo_label.addItems(["Edema (1)","No-realzante (2)","Realzante (3)"])
        self.combo_label.setCurrentIndex(2)
        lay_p.addWidget(self.combo_label, 1, 1)
        lay_p.addWidget(QLabel("Epocas:"), 2, 0)
        self.spin_ep = QSpinBox()
        self.spin_ep.setRange(1000, 20000); self.spin_ep.setSingleStep(1000); self.spin_ep.setValue(8000)
        lay_p.addWidget(self.spin_ep, 2, 1)
        left.addWidget(grp_p)

        # Boton analizar
        self.btn_analizar = QPushButton("ANALIZAR")
        self.btn_analizar.setObjectName("btnAnalizar")
        self.btn_analizar.setFixedWidth(300)
        self.btn_analizar.setEnabled(False)
        self.btn_analizar.clicked.connect(self._analizar)
        left.addWidget(self.btn_analizar)

        self.progress = QProgressBar(); self.progress.setFixedWidth(300); self.progress.setValue(0)
        left.addWidget(self.progress)
        self.lbl_prog = QLabel("Listo")
        self.lbl_prog.setStyleSheet("color: #6080a0; font-size: 11px;")
        self.lbl_prog.setWordWrap(True); self.lbl_prog.setFixedWidth(300)
        left.addWidget(self.lbl_prog)

        # Resultado
        grp_r = QGroupBox("Resultado Clinico"); grp_r.setFixedWidth(300)
        lay_r = QVBoxLayout(grp_r)
        self.lbl_riesgo = QLabel("---")
        self.lbl_riesgo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_riesgo.setStyleSheet("color: #6080a0; font-size: 18px; font-weight: bold;")
        lay_r.addWidget(self.lbl_riesgo)
        self.lbl_recom = QLabel("")
        self.lbl_recom.setStyleSheet("color: #a0c0e0; font-size: 11px;")
        self.lbl_recom.setWordWrap(True); self.lbl_recom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_r.addWidget(self.lbl_recom)

        self.lbl_D     = QLabel("D       : ---")
        self.lbl_rho   = QLabel("rho     : ---")
        self.lbl_div   = QLabel("D/rho   : ---")
        self.lbl_v     = QLabel("v_frente: ---")
        self.lbl_pde   = QLabel("PDE     : ---")
        self.lbl_nrmse = QLabel("NRMSE   : ---")
        self.lbl_pils  = QLabel("Pilares : ---")
        self.lbl_cob   = QLabel("Cob     : ---")
        for lb in [self.lbl_D,self.lbl_rho,self.lbl_div,self.lbl_v,
                   self.lbl_pde,self.lbl_nrmse,self.lbl_pils,self.lbl_cob]:
            lb.setStyleSheet("color: #80c0a0; font-size: 11px;")
            lay_r.addWidget(lb)
        left.addWidget(grp_r)

        # Boton PDF
        self.btn_pdf = QPushButton("Guardar Reporte PDF")
        self.btn_pdf.setObjectName("btnPDF")
        self.btn_pdf.setFixedWidth(300)
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self._guardar_pdf)
        left.addWidget(self.btn_pdf)

        left.addStretch()

        # Panel derecho
        right = QVBoxLayout(); right.setSpacing(6)
        body.addLayout(right, stretch=1)

        if VTK_OK:
            grp_v = QGroupBox("Visor 3D - Tumor")
            lay_v = QVBoxLayout(grp_v)
            self.vtk_widget = QVTKRenderWindowInteractor()
            self.vtk_widget.setMinimumHeight(500)
            lay_v.addWidget(self.vtk_widget)
            self._init_vtk()
            right.addWidget(grp_v, stretch=2)
        else:
            lbl_nov = QLabel("VTK no disponible\npip install vtk")
            lbl_nov.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_nov.setStyleSheet("color: #6080a0; font-size: 14px;")
            right.addWidget(lbl_nov, stretch=2)

        grp_log = QGroupBox("Log")
        lay_log = QVBoxLayout(grp_log)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(180)
        lay_log.addWidget(self.log)
        right.addWidget(grp_log, stretch=1)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage("NEXUS Medical listo  |  Abra un dataset BraTS para comenzar")

    def _init_vtk(self):
        self.ren3d = vtk.vtkRenderer()
        self.ren3d.SetBackground(0.05, 0.05, 0.10)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren3d)
        self.vtk_widget.Initialize()
        self.vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        t = vtk.vtkTextActor()
        t.SetInput("Abra un dataset BraTS para visualizar")
        t.GetTextProperty().SetFontSize(14); t.GetTextProperty().SetColor(0.4,0.6,0.8)
        t.SetPosition(20,20); self.ren3d.AddViewProp(t)

    def _log(self, msg):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _abrir_dataset(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta BraTS")
        if carpeta:
            self._cargar_dataset(carpeta)

    def _cargar_dataset(self, carpeta):
        if not NIB_OK: self._log("ERROR: nibabel no instalado"); return
        imgs_dir = os.path.join(carpeta, 'imagesTr')
        lbls_dir = os.path.join(carpeta, 'labelsTr')
        if not os.path.exists(imgs_dir): self._log(f"No se encontro imagesTr en {carpeta}"); return
        imgs = sorted([f.replace('.nii.gz','') for f in os.listdir(imgs_dir)
                       if f.endswith('.nii.gz') and not f.startswith('._')])
        lbls = sorted([f.replace('.nii.gz','') for f in os.listdir(lbls_dir)
                       if f.endswith('.nii.gz') and not f.startswith('._')])
        casos = sorted(set(imgs) & set(lbls))
        self.base_dir = carpeta
        self.combo_caso.clear(); self.combo_caso.addItems(casos)
        self._log(f"Dataset: {len(casos)} casos en {os.path.basename(carpeta)}")
        self.status.showMessage(f"BraTS: {len(casos)} casos | {carpeta}")

    def _caso_seleccionado(self, caso):
        if not caso or not self.base_dir or not NIB_OK: return
        try:
            imgs_dir = os.path.join(self.base_dir, 'imagesTr')
            lbls_dir = os.path.join(self.base_dir, 'labelsTr')
            canal = self.combo_canal.currentIndex()
            label = self.combo_label.currentIndex() + 1

            vol_full = nib.load(os.path.join(imgs_dir,f"{caso}.nii.gz")).get_fdata().astype(np.float32)
            seg_full = nib.load(os.path.join(lbls_dir,f"{caso}.nii.gz")).get_fdata().astype(np.int32)
            sp_nib   = nib.load(os.path.join(imgs_dir,f"{caso}.nii.gz")).header.get_zooms()
            sp3 = (float(sp_nib[0]), float(sp_nib[1]), float(sp_nib[2]))

            vol   = vol_full[:,:,:,canal] if vol_full.ndim==4 else vol_full
            tumor = (seg_full==label).astype(np.uint8)
            n_vox = tumor.sum(); vol_cm3 = n_vox*sp3[0]*sp3[1]*sp3[2]/1000

            px_por_z = tumor.sum(axis=(0,1))
            z_c = int(np.argmax(px_por_z)); px_c = int(px_por_z[z_c])

            p1=np.percentile(vol[vol>0],1); p99=np.percentile(vol[vol>0],99)
            vol_norm = np.clip((vol-p1)/(p99-p1+1e-10),0,1).astype(np.float32)

            NX=NY=32
            u_raw = gaussian_filter(tumor[:,:,z_c].astype(float), sigma=2.0)
            if u_raw.max()>0: u_raw/=u_raw.max()
            u_t0  = np.clip(nd_zoom(u_raw,(NX/u_raw.shape[0],NY/u_raw.shape[1]),order=3),0,1).astype(np.float32)

            self.caso_data = {'caso':caso,'vol_norm':vol_norm,'tumor':tumor,
                              'sp3':sp3,'z_c':z_c,'px_c':px_c,'u_t0':u_t0,
                              'n_vox':n_vox,'vol_cm3':vol_cm3}

            self.lbl_info.setText(f"Shape: {vol.shape}\nTumor: {n_vox:,} vox = {vol_cm3:.1f} cm3\nCorte: z={z_c}  px={px_c}")
            self.btn_analizar.setEnabled(px_c>=150 and TORCH_OK)

            if px_c < 150:
                self._log(f"AVISO {caso}: solo {px_c}px (min 150)")
            else:
                self._log(f"Cargado {caso}: {vol_cm3:.1f}cm3  z={z_c}  {px_c}px")
                self.status.showMessage(f"{caso} | Tumor {vol_cm3:.1f}cm3 | Corte z={z_c}")

            if VTK_OK: self._render_tumor(vol_norm, tumor, sp3)

        except Exception as e:
            self._log(f"ERROR: {e}")

    def _render_tumor(self, vol_norm, tumor, sp3):
        self.ren3d.RemoveAllViewProps()

        def a_vtk(arr, sp):
            img = vtk.vtkImageData(); img.SetDimensions(*arr.shape)
            img.SetSpacing(*sp); img.SetOrigin(0,0,0)
            flat = numpy_support.numpy_to_vtk(
                arr.flatten(order='F').astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT)
            img.GetPointData().SetScalars(flat); return img

        # Cerebro semi-transparente (vol_norm > 0.15 = tejido cerebral)
        cerebro = (vol_norm > 0.15).astype(np.float32)
        vi_cer = a_vtk(cerebro, sp3)
        mc_cer = vtk.vtkMarchingCubes(); mc_cer.SetInputData(vi_cer); mc_cer.SetValue(0,0.5)
        mc_cer.ComputeNormalsOn(); mc_cer.Update()
        if mc_cer.GetOutput().GetNumberOfCells() > 0:
            sm_c = vtk.vtkSmoothPolyDataFilter(); sm_c.SetInputConnection(mc_cer.GetOutputPort())
            sm_c.SetNumberOfIterations(50); sm_c.Update()
            nr_c = vtk.vtkPolyDataNormals(); nr_c.SetInputConnection(sm_c.GetOutputPort()); nr_c.Update()
            mp_c = vtk.vtkPolyDataMapper(); mp_c.SetInputConnection(nr_c.GetOutputPort()); mp_c.ScalarVisibilityOff()
            ac_c = vtk.vtkActor(); ac_c.SetMapper(mp_c)
            ac_c.GetProperty().SetColor(0.85, 0.78, 0.65)  # color tejido cerebral
            ac_c.GetProperty().SetOpacity(0.12)
            ac_c.GetProperty().SetAmbient(0.3); ac_c.GetProperty().SetDiffuse(0.7)
            self.ren3d.AddActor(ac_c)

        # Tumor solido encima
        vi = a_vtk(tumor.astype(float), sp3)
        mc = vtk.vtkMarchingCubes(); mc.SetInputData(vi); mc.SetValue(0,0.5)
        mc.ComputeNormalsOn(); mc.Update()
        if mc.GetOutput().GetNumberOfCells() > 0:
            sm = vtk.vtkSmoothPolyDataFilter(); sm.SetInputConnection(mc.GetOutputPort())
            sm.SetNumberOfIterations(30); sm.Update()
            nr = vtk.vtkPolyDataNormals(); nr.SetInputConnection(sm.GetOutputPort()); nr.Update()
            mp = vtk.vtkPolyDataMapper(); mp.SetInputConnection(nr.GetOutputPort()); mp.ScalarVisibilityOff()
            ac = vtk.vtkActor(); ac.SetMapper(mp)
            ac.GetProperty().SetColor(0.9,0.3,0.1); ac.GetProperty().SetOpacity(0.92)
            ac.GetProperty().SetAmbient(0.15); ac.GetProperty().SetDiffuse(0.85)
            ac.GetProperty().SetSpecular(0.4); ac.GetProperty().SetSpecularPower(25)
            self.ren3d.AddActor(ac)

        vi2 = a_vtk(vol_norm, sp3)
        ol  = vtk.vtkOutlineFilter(); ol.SetInputData(vi2)
        mo  = vtk.vtkPolyDataMapper(); mo.SetInputConnection(ol.GetOutputPort())
        ao  = vtk.vtkActor(); ao.SetMapper(mo); ao.GetProperty().SetColor(0.2,0.2,0.35)
        self.ren3d.AddActor(ao)
        self.ren3d.ResetCamera()
        self.ren3d.GetActiveCamera().Elevation(20); self.ren3d.GetActiveCamera().Azimuth(30)
        self.vtk_widget.GetRenderWindow().Render()

    def _analizar(self):
        if not self.caso_data or not TORCH_OK: return
        epochs = self.spin_ep.value()
        self._log(f"Iniciando: {self.caso_data['caso']}  epochs={epochs}")
        self.btn_analizar.setEnabled(False); self.progress.setValue(0)
        self.worker = AnalisisWorker(self.caso_data, epochs=epochs)
        self.worker.progreso.connect(self._on_prog)
        self.worker.resultado.connect(self._on_res)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_prog(self, pct, msg):
        self.progress.setValue(pct); self.lbl_prog.setText(msg)
        self.status.showMessage(f"Analizando {pct}% | {msg}")

    def _on_res(self, res):
        self.progress.setValue(100); self.btn_analizar.setEnabled(True)
        self._ultimo_res = res   # guardar para PDF
        self.btn_pdf.setEnabled(True)
        riesgo = res['riesgo']
        self.lbl_riesgo.setText(f"RIESGO {riesgo}")
        if riesgo=="ALTO":
            self.lbl_riesgo.setStyleSheet("color:#ff4444;font-size:18px;font-weight:bold;background:#2a0000;border:2px solid #ff0000;border-radius:4px;padding:4px 12px;")
        elif riesgo=="MEDIO":
            self.lbl_riesgo.setStyleSheet("color:#ffaa00;font-size:18px;font-weight:bold;background:#2a1a00;border:2px solid #ffaa00;border-radius:4px;padding:4px 12px;")
        else:
            self.lbl_riesgo.setStyleSheet("color:#44ff44;font-size:18px;font-weight:bold;background:#002a00;border:2px solid #00ff00;border-radius:4px;padding:4px 12px;")
        self.lbl_recom.setText(res['recom'])
        self.lbl_D.setText    (f"D       : {res['D']:.6f} cm2/mes")
        self.lbl_rho.setText  (f"rho     : {res['rho']:.6f} 1/mes")
        self.lbl_div.setText  (f"D/rho   : {res['div']:.4f}")
        self.lbl_v.setText    (f"v_frente: {res['v']:.4f} cm/mes")
        self.lbl_pde.setText  (f"PDE     : {res['pde']:.2e}  {'OK' if res['pde']<1e-4 else 'REVISAR'}")
        self.lbl_nrmse.setText(f"NRMSE   : {res['nrmse']:.4f}  {'OK' if res['nrmse']<0.01 else 'REVISAR'}")
        self.lbl_pils.setText (f"Pilares : {res['pilares']}/5")
        self.lbl_cob.setText  (f"Cob: {res['cob_t0']:.1f}% -> {res['cob_pred']:.1f}%")
        caso = self.caso_data['caso'] if self.caso_data else "?"
        self._log(f"RESULTADO {caso}: D={res['D']:.5f} rho={res['rho']:.5f} D/rho={res['div']:.4f} {riesgo} {res['pilares']}/5")
        self.status.showMessage(f"Completo | {caso} | D/rho={res['div']:.4f} | {riesgo} | {res['pilares']}/5 pilares")

    def _guardar_pdf(self):
        if not self._ultimo_res or not self.caso_data:
            self._log("Sin resultado para exportar"); return
        try:
            from nexus_report import generar_reporte, REPORTLAB_OK
            if not REPORTLAB_OK:
                self._log("ERROR: pip install reportlab"); return
        except ImportError:
            self._log("ERROR: nexus_report.py no encontrado junto a nexus_ui.py")
            return

        import datetime
        datos = {
            'caso':      self.caso_data.get('caso', 'caso'),
            'fecha':     datetime.date.today().isoformat(),
            'D':         self._ultimo_res['D'],
            'rho':       self._ultimo_res['rho'],
            'div':       self._ultimo_res['div'],
            'v_frente':  self._ultimo_res['v'],
            'pde':       self._ultimo_res['pde'],
            'nrmse':     self._ultimo_res['nrmse'],
            'pilares':   self._ultimo_res['pilares'],
            'riesgo':    self._ultimo_res['riesgo'],
            'recom':     self._ultimo_res['recom'],
            'cob_t0':    self._ultimo_res.get('cob_t0', 0.0),
            'cob_pred':  self._ultimo_res.get('cob_pred', 0.0),
            'n_vox':     self.caso_data.get('n_vox', 0),
            'vol_cm3':   self.caso_data.get('vol_cm3', 0.0),
            'z_corte':   self.caso_data.get('z_c', 0),
            'modalidad': 'T1gd',
            'label':     'Realzante (3)',
            'epochs':    self.spin_ep.value(),
            'device':    str(DEVICE) if TORCH_OK else 'cpu',
        }

        from PySide6.QtWidgets import QFileDialog
        import os
        default_name = f"reporte_{datos['caso']}_{datos['fecha']}.pdf"
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar Reporte PDF",
            os.path.join(os.getcwd(), default_name),
            "PDF (*.pdf)")
        if not ruta:
            return

        self._log(f"Generando PDF: {os.path.basename(ruta)}...")
        self.status.showMessage("Generando PDF...")
        resultado = generar_reporte(datos, ruta_salida=ruta)
        if resultado:
            self._log(f"PDF guardado: {resultado}")
            self.status.showMessage(f"PDF guardado: {os.path.basename(resultado)}")
        else:
            self._log("ERROR generando PDF")
            self.status.showMessage("Error generando PDF")

    def _on_err(self, msg):
        self.progress.setValue(0); self.btn_analizar.setEnabled(True)
        self._log(f"ERROR: {msg[:300]}"); self.status.showMessage("Error - ver log")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate(); self.worker.wait()
        event.accept()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default=None)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("NEXUS Medical")
    ventana = NexusUI(base_dir=args.base)
    ventana.show()
    if VTK_OK:
        ventana.vtk_widget.Initialize()
        ventana.vtk_widget.Start()
    sys.exit(app.exec())
