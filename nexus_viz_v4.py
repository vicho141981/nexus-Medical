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

# nexus_viz_v4.py - NEXUS Medical - Visualizacion completa + overlay prediccion
import os, sys, argparse, math
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] pip install nibabel"); sys.exit(1)

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    print("[ERROR] pip install vtk"); sys.exit(1)

from scipy.ndimage import gaussian_filter, zoom as nd_zoom, center_of_mass

print("="*60)
print("  NEXUS Viz v4 - Completo + overlay prediccion Fisher-KPP")
print("="*60)

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str,
    default='F:/NEXUS/archivos nexus medico/Task01_BrainTumour')
parser.add_argument('--caso',   type=str,   default='BRATS_001')
parser.add_argument('--D',      type=float, default=0.001469)
parser.add_argument('--rho',    type=float, default=0.092261)
parser.add_argument('--riesgo', type=str,   default='BAJO')
args = parser.parse_args()

imgs_dir = os.path.join(args.base, 'imagesTr')
lbls_dir = os.path.join(args.base, 'labelsTr')

print(f"[Cargando] {args.caso}  D={args.D:.5f}  rho={args.rho:.5f}")
vol_full = nib.load(os.path.join(imgs_dir, f"{args.caso}.nii.gz")).get_fdata().astype(np.float32)
seg_full = nib.load(os.path.join(lbls_dir, f"{args.caso}.nii.gz")).get_fdata().astype(np.int32)
sp_nib   = nib.load(os.path.join(imgs_dir, f"{args.caso}.nii.gz")).header.get_zooms()
sp3 = (float(sp_nib[0]), float(sp_nib[1]), float(sp_nib[2]))

vol = vol_full[:,:,:,2]
p1  = np.percentile(vol[vol>0],1); p99 = np.percentile(vol[vol>0],99)
vol_norm = np.clip((vol-p1)/(p99-p1+1e-10),0,1).astype(np.float32)

edema     = (seg_full==1).astype(np.float32)
no_realz  = (seg_full==2).astype(np.float32)
realzante = (seg_full==3).astype(np.float32)
tumor_all = (seg_full>0).astype(np.float32)
cerebro   = (vol_norm>0.15).astype(np.float32)

# Centroide
cx,cy,cz = center_of_mass(tumor_all)
cx,cy,cz = int(round(cx)),int(round(cy)),int(round(cz))
nx,ny,nz = vol_norm.shape

# Prediccion Fisher-KPP en el corte del centroide
px_por_z = (seg_full>0).sum(axis=(0,1))
z_best   = int(np.argmax(px_por_z))
NX=NY=64  # mas resolucion para prediccion
slice_gt = tumor_all[:,:,z_best]
u_raw    = gaussian_filter(slice_gt, sigma=2.0)
if u_raw.max()>0: u_raw/=u_raw.max()
fx=NX/u_raw.shape[0]; fy=NY/u_raw.shape[1]
u_t0 = np.clip(nd_zoom(u_raw,(fx,fy),order=3),0,1).astype(np.float32)

# Simular prediccion futura con D y rho inferidos
D_f=args.D; rho_f=args.rho; T=0.6; NT=60; dx=1.0/(NX-1)
u_pred=u_t0.copy()
for _ in range(NT):
    up=np.pad(u_pred,1,mode='edge')
    lap=((up[2:,1:-1]-2*up[1:-1,1:-1]+up[:-2,1:-1])/dx**2+
         (up[1:-1,2:]-2*up[1:-1,1:-1]+up[1:-1,:-2])/dx**2)
    u_pred=np.clip(u_pred+T/NT*(D_f*lap+rho_f*u_pred*(1-u_pred)),0,1)

# Zona de invasion nueva
invasion = np.clip(u_pred - u_t0, 0, 1)
print(f"  Prediccion: cob_t0={np.mean(u_t0>0.1)*100:.1f}% -> cob_pred={np.mean(u_pred>0.1)*100:.1f}%")

def a_vtk(arr, sp):
    img=vtk.vtkImageData(); img.SetDimensions(*arr.shape)
    img.SetSpacing(*sp); img.SetOrigin(0,0,0)
    flat=numpy_support.numpy_to_vtk(arr.flatten(order='F').astype(np.float32),
                                     deep=True,array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(flat); return img

def mc_actor(arr,sp,color,opacity,smooth=20):
    vi=a_vtk(arr,sp)
    mc=vtk.vtkMarchingCubes(); mc.SetInputData(vi); mc.SetValue(0,0.5)
    mc.ComputeNormalsOn(); mc.Update()
    n=mc.GetOutput().GetNumberOfCells()
    if n==0: return None,0
    sm=vtk.vtkSmoothPolyDataFilter(); sm.SetInputConnection(mc.GetOutputPort())
    sm.SetNumberOfIterations(smooth); sm.Update()
    nr=vtk.vtkPolyDataNormals(); nr.SetInputConnection(sm.GetOutputPort()); nr.Update()
    mp=vtk.vtkPolyDataMapper(); mp.SetInputConnection(nr.GetOutputPort()); mp.ScalarVisibilityOff()
    ac=vtk.vtkActor(); ac.SetMapper(mp)
    ac.GetProperty().SetColor(*color); ac.GetProperty().SetOpacity(opacity)
    ac.GetProperty().SetAmbient(0.15); ac.GetProperty().SetDiffuse(0.85)
    ac.GetProperty().SetSpecular(0.4); ac.GetProperty().SetSpecularPower(25)
    return ac,n

def lut_grises():
    lut=vtk.vtkLookupTable(); lut.SetNumberOfTableValues(256); lut.SetRange(0.0,1.0)
    for i in range(256):
        v=i/255.0; lut.SetTableValue(i,v,v,v,1.0)
    lut.Build(); return lut

def slice_actor(vtk_img, axis, idx, lut):
    nx_,ny_,nz_=vtk_img.GetDimensions()
    pf=vtk.vtkImageDataGeometryFilter(); pf.SetInputData(vtk_img)
    if axis==0:   pf.SetExtent(idx,idx,0,ny_-1,0,nz_-1)
    elif axis==1: pf.SetExtent(0,nx_-1,idx,idx,0,nz_-1)
    else:         pf.SetExtent(0,nx_-1,0,ny_-1,idx,idx)
    pf.Update()
    mp=vtk.vtkPolyDataMapper(); mp.SetInputConnection(pf.GetOutputPort())
    mp.SetLookupTable(lut); mp.SetScalarRange(0.0,1.0); mp.ScalarVisibilityOn()
    ac=vtk.vtkActor(); ac.SetMapper(mp)
    return pf,ac

vtk_mri   = a_vtk(vol_norm, sp3)
lut_g     = lut_grises()

# Color riesgo
if args.riesgo=='ALTO':    c_riesgo=(1.0,0.2,0.2)
elif args.riesgo=='MEDIO': c_riesgo=(1.0,0.7,0.0)
else:                      c_riesgo=(0.2,1.0,0.3)

# ── 5 RENDERERS ────────────────────────────────────────────────────────────────
# Layout: [3D vol-render | Axial MRI] / [Sagital MRI | Prediccion Fisher-KPP]
ren3d  = vtk.vtkRenderer(); ren3d.SetViewport(0.0,0.5,0.5,1.0); ren3d.SetBackground(0.02,0.02,0.06)
ren_ax = vtk.vtkRenderer(); ren_ax.SetViewport(0.5,0.5,1.0,1.0); ren_ax.SetBackground(0.03,0.03,0.08)
ren_sg = vtk.vtkRenderer(); ren_sg.SetViewport(0.0,0.0,0.5,0.5); ren_sg.SetBackground(0.03,0.03,0.08)
ren_pr = vtk.vtkRenderer(); ren_pr.SetViewport(0.5,0.0,1.0,0.5); ren_pr.SetBackground(0.02,0.02,0.05)

# Vista 3D - Volume rendering
vol_mapper = vtk.vtkGPUVolumeRayCastMapper(); vol_mapper.SetInputData(vtk_mri)
c_tf=vtk.vtkColorTransferFunction()
c_tf.AddRGBPoint(0.00,0.00,0.00,0.00); c_tf.AddRGBPoint(0.15,0.30,0.30,0.45)
c_tf.AddRGBPoint(0.40,0.60,0.60,0.80); c_tf.AddRGBPoint(0.70,0.80,0.78,0.75)
c_tf.AddRGBPoint(1.00,1.00,0.95,0.85)
o_tf=vtk.vtkPiecewiseFunction()
o_tf.AddPoint(0.00,0.000); o_tf.AddPoint(0.10,0.000); o_tf.AddPoint(0.15,0.004)
o_tf.AddPoint(0.40,0.018); o_tf.AddPoint(0.70,0.035); o_tf.AddPoint(1.00,0.060)
vp=vtk.vtkVolumeProperty(); vp.SetColor(c_tf); vp.SetScalarOpacity(o_tf)
vp.ShadeOn(); vp.SetInterpolationTypeToLinear()
va=vtk.vtkVolume(); va.SetMapper(vol_mapper); va.SetProperty(vp)
ren3d.AddVolume(va)

# Tumores encima
ac_ede,n_ede=mc_actor(edema,    sp3,(1.0,0.90,0.0),0.40)
ac_nor,n_nor=mc_actor(no_realz, sp3,(1.0,0.45,0.0),0.70)
ac_rea,n_rea=mc_actor(realzante,sp3,(1.0,0.10,0.1),0.95)
for ac in [ac_ede,ac_nor,ac_rea]:
    if ac: ren3d.AddActor(ac)

# Cortes MRI con overlay tumor
lut_t=vtk.vtkLookupTable(); lut_t.SetNumberOfTableValues(256); lut_t.SetRange(0,1)
lut_t.SetTableValue(0,0,0,0,0)
for i in range(1,256): v=i/255.0; lut_t.SetTableValue(i,1.0,0.3,0.0,min(0.9,v*2))
lut_t.Build()
vtk_seg=a_vtk((seg_full>0).astype(np.float32),sp3)

_,ac_ax_m=slice_actor(vtk_mri,2,cz,lut_g); _,ac_ax_t=slice_actor(vtk_seg,2,cz,lut_t)
_,ac_sg_m=slice_actor(vtk_mri,0,cx,lut_g); _,ac_sg_t=slice_actor(vtk_seg,0,cx,lut_t)
ren_ax.AddActor(ac_ax_m); ren_ax.AddActor(ac_ax_t)
ren_sg.AddActor(ac_sg_m); ren_sg.AddActor(ac_sg_t)

# Panel prediccion — imagen RGB 2D (Fisher-KPP)
# [FIX] Reemplaza loop O(H*W) por numpy_to_vtk vectorizado
# Codificacion de color:
#   Rojo   = tumor actual (u_t0)
#   Verde  = zona de invasion nueva (u_pred - u_t0, clipeada)
#   Azul   = prediccion futura total (u_pred)
H, W = u_t0.shape
rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgb[:, :, 0] = np.clip(u_t0          * 255, 0, 255).astype(np.uint8)   # rojo
rgb[:, :, 1] = np.clip(invasion * 2.0 * 255, 0, 255).astype(np.uint8)  # verde
rgb[:, :, 2] = np.clip(u_pred  * 0.8 * 255, 0, 255).astype(np.uint8)   # azul

# [FIX] vtkImageData con dimensiones (W, H, 1) — ejes VTK: X=columnas, Y=filas
# El array se flatten en orden C (fila mayor) y se pasa directo a VTK
img_rgb = vtk.vtkImageData()
img_rgb.SetDimensions(W, H, 1)
img_rgb.SetSpacing(4.0, 4.0, 1.0)
img_rgb.SetOrigin(0.0, 0.0, 0.0)
# numpy_to_vtk espera array (N, ncomp) — reshape a (W*H, 3)
flat_rgb = rgb.reshape(-1, 3)
vtk_arr  = numpy_support.numpy_to_vtk(flat_rgb, deep=True,
                                       array_type=vtk.VTK_UNSIGNED_CHAR)
vtk_arr.SetNumberOfComponents(3)
img_rgb.GetPointData().SetScalars(vtk_arr)

# [FIX] vtkImageActor renderiza imagen 2D directamente sin Marching Cubes
img_actor = vtk.vtkImageActor()
img_actor.GetMapper().SetInputData(img_rgb)
img_actor.SetDisplayExtent(0, W-1, 0, H-1, 0, 0)
ren_pr.AddActor(img_actor)
ren_pr.ResetCamera()
ren_pr.GetActiveCamera().ParallelProjectionOn()

# Leyenda de colores en el panel prediccion
def color_legend(ren, items, x=12, y_start=30, dy=18):
    """items = list of (r,g,b, texto)"""
    for i, (r, g, b, txt) in enumerate(items):
        sq = vtk.vtkTextActor()
        sq.SetInput("■")
        sq.GetTextProperty().SetFontSize(14)
        sq.GetTextProperty().SetColor(r, g, b)
        sq.SetPosition(x, y_start + i*dy)
        ren.AddViewProp(sq)
        tl = vtk.vtkTextActor()
        tl.SetInput(txt)
        tl.GetTextProperty().SetFontSize(11)
        tl.GetTextProperty().SetColor(0.8, 0.85, 0.9)
        tl.SetPosition(x+18, y_start + i*dy)
        ren.AddViewProp(tl)

color_legend(ren_pr, [
    (1.0, 0.2, 0.2, "Tumor actual (t0)"),
    (0.2, 1.0, 0.2, "Zona invasion nueva"),
    (0.2, 0.5, 1.0, "Prediccion futura"),
])

# Labels
def lbl(ren,txt,color=(0.5,0.85,1.0)):
    t=vtk.vtkTextActor(); t.SetInput(txt)
    t.GetTextProperty().SetFontSize(13); t.GetTextProperty().SetColor(*color)
    t.GetTextProperty().BoldOn(); t.SetPosition(8,8); ren.AddViewProp(t)

v_f=2*math.sqrt(args.D*args.rho)
lbl(ren3d,f"3D Vol.Render | {args.caso} | Edema+NoRealz+Realzante")
lbl(ren_ax,f"AXIAL z={cz} | MRI T1gd + tumor (naranja)")
lbl(ren_sg,f"SAGITAL x={cx} | MRI T1gd + tumor (naranja)")
lbl(ren_pr,f"PREDICCION +30d | D={args.D:.4f} rho={args.rho:.4f} v={v_f:.4f} cm/mes | RIESGO {args.riesgo}",
    c_riesgo)

# [FIX] ren_pr excluido — ya tiene ParallelProjection + ResetCamera propio
for r in [ren3d,ren_ax,ren_sg]: r.ResetCamera()
ren3d.GetActiveCamera().Elevation(15); ren3d.GetActiveCamera().Azimuth(25)

win=vtk.vtkRenderWindow(); win.SetSize(1400,900)
win.SetWindowName(f"NEXUS Medical - {args.caso} - Visualizacion Completa")
for r in [ren3d,ren_ax,ren_sg,ren_pr]: win.AddRenderer(r)

ix=vtk.vtkRenderWindowInteractor(); ix.SetRenderWindow(win)
ix.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
print(f"  q=salir | drag=rotar | scroll=zoom")
print(f"  Panel 4: prediccion a +30 dias | zona invasion en {c_riesgo}")
win.Render(); ix.Start()
print("[v4] OK")
