# Instrucciones para subir NEXUS Medical a GitHub

## Paso 1 — Copiar archivos a F:\NEXUS\

Copiar todos los archivos de este ZIP a `F:\NEXUS\`

## Paso 2 — Abrir terminal en F:\NEXUS\

```powershell
cd F:\NEXUS
```

## Paso 3 — Inicializar git y subir

```powershell
git init
git add nexus_ui.py nexus_report.py nexus_brats_pipeline_v4.py nexus_validacion_v7.py nexus_viz_v1.py nexus_viz_v2.py nexus_viz_v3.py nexus_viz_v4.py train_sensor_fusion_v2.py README.md LICENSE .gitignore nexus_paper.docx
git commit -m "NEXUS Medical v1.0 — Fisher-KPP PINN for brain tumor invasiveness estimation"
git branch -M main
git remote add origin https://github.com/vicho141981/nexus-Medical.git
git push -u origin main
```

## Paso 4 — Verificar en GitHub

Ir a: https://github.com/vicho141981/nexus-Medical

## Archivos incluidos

| Archivo | Descripcion |
|---------|-------------|
| nexus_ui.py | UI principal PySide6 + VTK + PINN |
| nexus_report.py | Generador PDF clinico |
| nexus_brats_pipeline_v4.py | Pipeline batch BraTS 20 casos |
| nexus_validacion_v7.py | Validacion cientifica vs Zhang 2024 |
| nexus_viz_v1..v4.py | Visualizadores VTK standalone |
| train_sensor_fusion_v2.py | Modulo SensorFusion MultiheadAttention |
| nexus_paper.docx | Paper cientifico completo |
| README.md | Documentacion GitHub |
| LICENSE | Apache 2.0 |
| .gitignore | Excluye dataset y resultados locales |

## Notas importantes

- El dataset BraTS (484 casos .nii.gz) NO se sube — son varios GB
- Los resultados de runs_medico/ NO se suben — son locales
- El paper tiene placeholders [Apellido] y [email] — completar antes de subir
