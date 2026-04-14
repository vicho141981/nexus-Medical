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

# nexus_viz_v2.py - NEXUS Medical - 4 paneles 3D + Axial + Sagital + Coronal
import os, sys, argparse
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

print("="*60)
print("  NEXUS Viz v2 - 4 paneles ortogonales")
print("="*60)

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str,
    default='F:/NEXUS/archivos nexus medico/Task01_BrainTumour')
parser.add_argument('--caso', type=str, default='BRATS_001')
args = parser.parse_args()

imgs_dir = os.path.join(args.base, 'imagesTr')
lbls_dir = os.path.join(args.base, 'labelsTr')

print(f"[Cargando] {args.caso}...")
vol_full = nib.load(os.path.join(imgs_dir, f"{args.caso}.nii.gz")).get_fdata().astype(np.float32)
seg_full = nib.load(os.path.join(lbls_dir, f"{args.caso}.nii.gz")).get_fdata().astype(np.int32)
sp_nib   = nib.load(os.path.join(imgs_dir, f"{args.caso}.nii.gz")).header.get_zooms()
sp3 = (float(sp_nib[0]), float(sp_nib[1]), float(sp_nib[2]))

vol = vol_full[:,:,:,2]
p1  = np.percentile(vol[vol>0], 1); p99 = np.percentile(vol[vol>0], 99)
vol_norm = np.clip((vol-p1)/(p99-p1+1e-10), 0, 1).astype(np.float32)

# Mascara tumoral (todas las regiones)
tumor_all = (seg_full > 0).astype(np.float32)
realzante = (seg_full == 3).astype(np.float32)

# Centroide del tumor
from scipy.ndimage import center_of_mass
cx, cy, cz = center_of_mass(tumor_all)
cx, cy, cz = int(round(cx)), int(round(cy)), int(round(cz))
nx, ny, nz = vol_norm.shape
print(f"  Shape: {vol_norm.shape}  Centroide: ({cx},{cy},{cz})")

def a_vtk(arr, sp):
    img = vtk.vtkImageData()
    img.SetDimensions(*arr.shape)
    img.SetSpacing(*sp); img.SetOrigin(0,0,0)
    flat = numpy_support.numpy_to_vtk(
        arr.flatten(order='F').astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(flat)
    return img

vtk_mri   = a_vtk(vol_norm, sp3)
vtk_tumor = a_vtk(tumor_all, sp3)

# LUT escala de grises para MRI
def lut_grises():
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256); lut.SetRange(0.0, 1.0)
    for i in range(256):
        v = i/255.0
        lut.SetTableValue(i, v, v, v, 1.0)
    lut.Build()
    return lut

# LUT para tumor (overlay coloreado)
def lut_tumor():
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256); lut.SetRange(0.0, 1.0)
    lut.SetTableValue(0, 0,0,0,0)  # transparente en 0
    for i in range(1,256):
        v = i/255.0
        lut.SetTableValue(i, 1.0, 0.2*v, 0.0, 0.7)  # naranja-rojo
    lut.Build()
    return lut

lut_g = lut_grises()
lut_t = lut_tumor()

# Corte 2D usando vtkImageDataGeometryFilter
def make_slice_actor(vtk_img, axis, idx, lut):
    nx_,ny_,nz_ = vtk_img.GetDimensions()
    pf = vtk.vtkImageDataGeometryFilter()
    pf.SetInputData(vtk_img)
    if axis == 0:    # Sagital X
        pf.SetExtent(idx,idx, 0,ny_-1, 0,nz_-1)
    elif axis == 1:  # Coronal Y
        pf.SetExtent(0,nx_-1, idx,idx, 0,nz_-1)
    else:            # Axial Z
        pf.SetExtent(0,nx_-1, 0,ny_-1, idx,idx)
    pf.Update()
    mp = vtk.vtkPolyDataMapper()
    mp.SetInputConnection(pf.GetOutputPort())
    mp.SetLookupTable(lut); mp.SetScalarRange(0.0,1.0)
    mp.ScalarVisibilityOn()
    ac = vtk.vtkActor(); ac.SetMapper(mp)
    return pf, ac

# Actor 3D del tumor
def mc_actor_3d(arr, sp, color, opacity):
    vi = a_vtk(arr, sp)
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vi); mc.SetValue(0,0.5); mc.ComputeNormalsOn(); mc.Update()
    if mc.GetOutput().GetNumberOfCells()==0: return None
    sm = vtk.vtkSmoothPolyDataFilter()
    sm.SetInputConnection(mc.GetOutputPort()); sm.SetNumberOfIterations(30); sm.Update()
    nr = vtk.vtkPolyDataNormals(); nr.SetInputConnection(sm.GetOutputPort()); nr.Update()
    mp = vtk.vtkPolyDataMapper(); mp.SetInputConnection(nr.GetOutputPort()); mp.ScalarVisibilityOff()
    ac = vtk.vtkActor(); ac.SetMapper(mp)
    ac.GetProperty().SetColor(*color); ac.GetProperty().SetOpacity(opacity)
    ac.GetProperty().SetAmbient(0.15); ac.GetProperty().SetDiffuse(0.85)
    ac.GetProperty().SetSpecular(0.3); ac.GetProperty().SetSpecularPower(20)
    return ac

# 4 renderers
def make_ren(vp, bg=(0.05,0.05,0.10)):
    r = vtk.vtkRenderer(); r.SetBackground(*bg); r.SetViewport(*vp); return r

ren3d  = make_ren((0.0, 0.5, 0.5, 1.0))
ren_ax = make_ren((0.5, 0.5, 1.0, 1.0), (0.03,0.03,0.08))
ren_sg = make_ren((0.0, 0.0, 0.5, 0.5), (0.03,0.03,0.08))
ren_co = make_ren((0.5, 0.0, 1.0, 0.5), (0.03,0.03,0.08))

# Vista 3D - cerebro + tumor
cerebro_arr = (vol_norm > 0.15).astype(np.float32)
ac_cer = mc_actor_3d(cerebro_arr, sp3, (0.85,0.78,0.65), 0.10)
ac_rea = mc_actor_3d(realzante,   sp3, (1.0, 0.2,  0.1), 0.92)
ac_all = mc_actor_3d(tumor_all,   sp3, (1.0, 0.5,  0.0), 0.50)

if ac_cer: ren3d.AddActor(ac_cer)
if ac_all: ren3d.AddActor(ac_all)
if ac_rea: ren3d.AddActor(ac_rea)

ol = vtk.vtkOutlineFilter(); ol.SetInputData(vtk_mri)
mo = vtk.vtkPolyDataMapper(); mo.SetInputConnection(ol.GetOutputPort())
ao = vtk.vtkActor(); ao.SetMapper(mo); ao.GetProperty().SetColor(0.2,0.2,0.35)
ren3d.AddActor(ao)

# Cortes 2D — MRI de fondo + overlay tumor
pf_ax_m, ac_ax_m = make_slice_actor(vtk_mri,   2, cz, lut_g)
pf_ax_t, ac_ax_t = make_slice_actor(vtk_tumor, 2, cz, lut_t)
pf_sg_m, ac_sg_m = make_slice_actor(vtk_mri,   0, cx, lut_g)
pf_sg_t, ac_sg_t = make_slice_actor(vtk_tumor, 0, cx, lut_t)
pf_co_m, ac_co_m = make_slice_actor(vtk_mri,   1, cy, lut_g)
pf_co_t, ac_co_t = make_slice_actor(vtk_tumor, 1, cy, lut_t)

ren_ax.AddActor(ac_ax_m); ren_ax.AddActor(ac_ax_t)
ren_sg.AddActor(ac_sg_m); ren_sg.AddActor(ac_sg_t)
ren_co.AddActor(ac_co_m); ren_co.AddActor(ac_co_t)

def lbl(ren, txt, color=(0.5,0.85,1.0)):
    t = vtk.vtkTextActor(); t.SetInput(txt)
    t.GetTextProperty().SetFontSize(13)
    t.GetTextProperty().SetColor(*color)
    t.GetTextProperty().BoldOn(); t.SetPosition(8,8)
    ren.AddViewProp(t)

lbl(ren3d,  f"3D - {args.caso}  (rotar: drag  zoom: scroll)")
lbl(ren_ax, f"AXIAL  z={cz}")
lbl(ren_sg, f"SAGITAL  x={cx}")
lbl(ren_co, f"CORONAL  y={cy}")

for r in [ren3d, ren_ax, ren_sg, ren_co]:
    r.ResetCamera()
ren3d.GetActiveCamera().Elevation(20); ren3d.GetActiveCamera().Azimuth(30)

# Ventana
win = vtk.vtkRenderWindow()
win.SetSize(1280, 800)
win.SetWindowName(f"NEXUS Viz v2 - {args.caso} - 4 paneles")
for r in [ren3d, ren_ax, ren_sg, ren_co]: win.AddRenderer(r)

ix = vtk.vtkRenderWindowInteractor()
ix.SetRenderWindow(win)
ix.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

print(f"[Render] 4 paneles  centroide=({cx},{cy},{cz})  q=salir")
win.Render(); ix.Start()
print("[v2] OK")
