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

# nexus_viz_v3.py - NEXUS Medical - Volume rendering translucido del cerebro
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
print("  NEXUS Viz v3 - Volume rendering cerebro translucido")
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

# T1gd canal 2 - mejor para ver tumor
vol = vol_full[:,:,:,2]
p1  = np.percentile(vol[vol>0], 1); p99 = np.percentile(vol[vol>0], 99)
vol_norm = np.clip((vol-p1)/(p99-p1+1e-10), 0, 1).astype(np.float32)

edema     = (seg_full == 1).astype(np.float32)
no_realz  = (seg_full == 2).astype(np.float32)
realzante = (seg_full == 3).astype(np.float32)

def a_vtk(arr, sp):
    img = vtk.vtkImageData()
    img.SetDimensions(*arr.shape)
    img.SetSpacing(*sp); img.SetOrigin(0,0,0)
    flat = numpy_support.numpy_to_vtk(
        arr.flatten(order='F').astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(flat)
    return img

vtk_vol = a_vtk(vol_norm, sp3)

# Volume rendering con GPU ray casting
vol_mapper = vtk.vtkGPUVolumeRayCastMapper()
vol_mapper.SetInputData(vtk_vol)

# Color transfer function — cerebro en azul-gris translucido
color_tf = vtk.vtkColorTransferFunction()
color_tf.AddRGBPoint(0.00, 0.00, 0.00, 0.00)  # fondo negro
color_tf.AddRGBPoint(0.10, 0.10, 0.10, 0.15)  # materia blanca tenue
color_tf.AddRGBPoint(0.25, 0.50, 0.50, 0.65)  # materia gris
color_tf.AddRGBPoint(0.50, 0.70, 0.70, 0.85)  # tejido cerebral
color_tf.AddRGBPoint(0.80, 0.90, 0.88, 0.80)  # zona brillante
color_tf.AddRGBPoint(1.00, 1.00, 0.95, 0.85)  # maxima intensidad

# Opacidad — muy translucido para ver el interior
opacity_tf = vtk.vtkPiecewiseFunction()
opacity_tf.AddPoint(0.00, 0.000)  # fondo invisible
opacity_tf.AddPoint(0.10, 0.000)  # ruido invisible
opacity_tf.AddPoint(0.15, 0.005)  # inicio cerebro casi invisible
opacity_tf.AddPoint(0.30, 0.020)  # tejido muy translucido
opacity_tf.AddPoint(0.50, 0.035)  # cerebro principal
opacity_tf.AddPoint(0.80, 0.050)  # zona brillante
opacity_tf.AddPoint(1.00, 0.070)  # maxima intensidad

vol_prop = vtk.vtkVolumeProperty()
vol_prop.SetColor(color_tf)
vol_prop.SetScalarOpacity(opacity_tf)
vol_prop.ShadeOn()
vol_prop.SetInterpolationTypeToLinear()
vol_prop.SetAmbient(0.2); vol_prop.SetDiffuse(0.8); vol_prop.SetSpecular(0.3)

volume_actor = vtk.vtkVolume()
volume_actor.SetMapper(vol_mapper)
volume_actor.SetProperty(vol_prop)

# Tumor en Marching Cubes encima del volume rendering
def mc_actor(arr, sp, color, opacity, smooth=20):
    vi = a_vtk(arr, sp)
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vi); mc.SetValue(0,0.5); mc.ComputeNormalsOn(); mc.Update()
    n = mc.GetOutput().GetNumberOfCells()
    if n == 0: return None, 0
    sm = vtk.vtkSmoothPolyDataFilter()
    sm.SetInputConnection(mc.GetOutputPort()); sm.SetNumberOfIterations(smooth); sm.Update()
    nr = vtk.vtkPolyDataNormals(); nr.SetInputConnection(sm.GetOutputPort()); nr.Update()
    mp = vtk.vtkPolyDataMapper(); mp.SetInputConnection(nr.GetOutputPort()); mp.ScalarVisibilityOff()
    ac = vtk.vtkActor(); ac.SetMapper(mp)
    ac.GetProperty().SetColor(*color); ac.GetProperty().SetOpacity(opacity)
    ac.GetProperty().SetAmbient(0.15); ac.GetProperty().SetDiffuse(0.85)
    ac.GetProperty().SetSpecular(0.5); ac.GetProperty().SetSpecularPower(30)
    return ac, n

ren = vtk.vtkRenderer()
ren.SetBackground(0.02, 0.02, 0.06)  # casi negro, toque azul

# Volume rendering del cerebro
ren.AddVolume(volume_actor)

# 3 capas tumorales encima
ac_ede, n_ede = mc_actor(edema,     sp3, (1.0, 0.90, 0.0), 0.45)
ac_nor, n_nor = mc_actor(no_realz,  sp3, (1.0, 0.45, 0.0), 0.75)
ac_rea, n_rea = mc_actor(realzante, sp3, (1.0, 0.10, 0.1), 0.95)

for ac in [ac_ede, ac_nor, ac_rea]:
    if ac: ren.AddActor(ac)

# Outline del volumen
ol = vtk.vtkOutlineFilter(); ol.SetInputData(vtk_vol)
mo = vtk.vtkPolyDataMapper(); mo.SetInputConnection(ol.GetOutputPort())
ao = vtk.vtkActor(); ao.SetMapper(mo); ao.GetProperty().SetColor(0.15,0.15,0.25); ao.GetProperty().SetOpacity(0.4)
ren.AddActor(ao)

# Labels
def lbl(ren, txt, color, y):
    t = vtk.vtkTextActor(); t.SetInput(txt)
    t.GetTextProperty().SetFontSize(13)
    t.GetTextProperty().SetColor(*color)
    t.GetTextProperty().BoldOn(); t.SetPosition(10, y)
    ren.AddViewProp(t)

lbl(ren, f"Realzante    {n_rea:,} vox",  (1.0, 0.3, 0.2), 70)
lbl(ren, f"No-realzante {n_nor:,} vox",  (1.0, 0.6, 0.1), 50)
lbl(ren, f"Edema        {n_ede:,} vox",  (1.0, 0.9, 0.0), 30)
lbl(ren, f"{args.caso}  |  q=salir  |  scroll=zoom", (0.5,0.7,1.0), 10)

ren.ResetCamera()
cam = ren.GetActiveCamera()
cam.Elevation(15); cam.Azimuth(25)
ren.ResetCameraClippingRange()

win = vtk.vtkRenderWindow()
win.SetSize(1100, 800)
win.SetWindowName(f"NEXUS Viz v3 - {args.caso} - Volume Rendering")
win.AddRenderer(ren)

ix = vtk.vtkRenderWindowInteractor()
ix.SetRenderWindow(win)
ix.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

print(f"[Render] Volume rendering  q=salir")
win.Render(); ix.Start()
print("[v3] OK")
