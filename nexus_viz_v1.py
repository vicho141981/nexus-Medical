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

# nexus_viz_v1.py - NEXUS Medical - Visualizacion 3 colores por region tumoral
# Edema=amarillo | No-realzante=naranja | Realzante=rojo brillante
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
print("  NEXUS Viz v1 - 3 regiones tumorales en color")
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

vol = vol_full[:,:,:,2]  # T1gd
p1  = np.percentile(vol[vol>0], 1)
p99 = np.percentile(vol[vol>0], 99)
vol_norm = np.clip((vol-p1)/(p99-p1+1e-10), 0, 1).astype(np.float32)

# Mascaras por label
edema     = (seg_full == 1).astype(np.float32)
no_realz  = (seg_full == 2).astype(np.float32)
realzante = (seg_full == 3).astype(np.float32)
cerebro   = (vol_norm > 0.15).astype(np.float32)

print(f"  Edema      : {edema.sum():,} vox")
print(f"  No-realz   : {no_realz.sum():,} vox")
print(f"  Realzante  : {realzante.sum():,} vox")
print(f"  Cerebro    : {cerebro.sum():,} vox")

def a_vtk(arr, sp):
    img = vtk.vtkImageData()
    img.SetDimensions(*arr.shape)
    img.SetSpacing(*sp); img.SetOrigin(0,0,0)
    flat = numpy_support.numpy_to_vtk(
        arr.flatten(order='F').astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(flat)
    return img

def mc_actor(arr, sp, thresh, color, opacity, smooth=30):
    vi = a_vtk(arr, sp)
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vi); mc.SetValue(0, thresh); mc.ComputeNormalsOn(); mc.Update()
    n = mc.GetOutput().GetNumberOfCells()
    if n == 0: return None, 0
    sm = vtk.vtkSmoothPolyDataFilter()
    sm.SetInputConnection(mc.GetOutputPort())
    sm.SetNumberOfIterations(smooth); sm.Update()
    nr = vtk.vtkPolyDataNormals(); nr.SetInputConnection(sm.GetOutputPort()); nr.Update()
    mp = vtk.vtkPolyDataMapper(); mp.SetInputConnection(nr.GetOutputPort()); mp.ScalarVisibilityOff()
    ac = vtk.vtkActor(); ac.SetMapper(mp)
    ac.GetProperty().SetColor(*color)
    ac.GetProperty().SetOpacity(opacity)
    ac.GetProperty().SetAmbient(0.15); ac.GetProperty().SetDiffuse(0.85)
    ac.GetProperty().SetSpecular(0.4); ac.GetProperty().SetSpecularPower(20)
    return ac, n

ren = vtk.vtkRenderer()
ren.SetBackground(0.05, 0.05, 0.10)

# 1. Cerebro translucido
ac_cer, n_cer = mc_actor(cerebro,   sp3, 0.5, (0.85,0.78,0.65), 0.08, smooth=60)
# 2. Edema - amarillo
ac_ede, n_ede = mc_actor(edema,     sp3, 0.5, (1.0, 0.85, 0.0), 0.55, smooth=20)
# 3. No-realzante - naranja
ac_nor, n_nor = mc_actor(no_realz,  sp3, 0.5, (1.0, 0.45, 0.0), 0.80, smooth=20)
# 4. Realzante - rojo brillante
ac_rea, n_rea = mc_actor(realzante, sp3, 0.5, (1.0, 0.15, 0.1), 0.95, smooth=20)

for ac in [ac_cer, ac_ede, ac_nor, ac_rea]:
    if ac: ren.AddActor(ac)

# Outline
vi2 = a_vtk(vol_norm, sp3)
ol  = vtk.vtkOutlineFilter(); ol.SetInputData(vi2)
mo  = vtk.vtkPolyDataMapper(); mo.SetInputConnection(ol.GetOutputPort())
ao  = vtk.vtkActor(); ao.SetMapper(mo); ao.GetProperty().SetColor(0.2,0.2,0.3)
ren.AddActor(ao)

# Leyenda
def add_label(ren, txt, color, pos_y):
    t = vtk.vtkTextActor()
    t.SetInput(txt)
    t.GetTextProperty().SetFontSize(14)
    t.GetTextProperty().SetColor(*color)
    t.GetTextProperty().BoldOn()
    t.SetPosition(10, pos_y)
    ren.AddViewProp(t)

add_label(ren, f"Realzante   : {n_rea:,} vox",  (1.0, 0.3, 0.2), 70)
add_label(ren, f"No-realzante: {n_nor:,} vox",  (1.0, 0.6, 0.1), 50)
add_label(ren, f"Edema       : {n_ede:,} vox",  (1.0, 0.9, 0.0), 30)
add_label(ren, f"Caso: {args.caso}  |  q=salir", (0.5, 0.7, 1.0), 10)

ren.ResetCamera()
ren.GetActiveCamera().Elevation(20)
ren.GetActiveCamera().Azimuth(30)

win = vtk.vtkRenderWindow()
win.SetSize(1100, 800)
win.SetWindowName(f"NEXUS Viz v1 - {args.caso} - 3 regiones tumorales")
win.AddRenderer(ren)

ix = vtk.vtkRenderWindowInteractor()
ix.SetRenderWindow(win)
ix.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

print(f"\n[Render] Abriendo visor...")
print(f"  Cerebro: {n_cer:,} tri  Edema: {n_ede:,} tri  No-realz: {n_nor:,} tri  Realzante: {n_rea:,} tri")
print(f"  Controles: click+drag=rotar  scroll=zoom  q=salir")

win.Render()
ix.Start()
print("[v1] OK")
