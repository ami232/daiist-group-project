# MaintAI · Plan de trabajo del equipo

Guía interna para trabajar en paralelo sin pisarnos. El plan completo está en el master plan; aquí solo lo que cada persona necesita para empezar. Las decisiones y supuestos del business case están en `REPORT.md` (aprobados por Manuela, pendientes de confirmación del resto del equipo).

Entrega: **lunes 30 nov, 23:59**. App: https://maintai-238a.onrender.com

## 1. Cómo empezar (cada persona, una vez)

```bash
git clone https://github.com/manueladominguez/daiist-group-project.git
cd daiist-group-project
uv sync
cp .env.example .env          # pega TU propia GEMINI_API_KEY en .env (nunca se sube)
uv run python app.py          # http://127.0.0.1:7860, con el banner "Development version"
```

Cada persona usa su **propia clave de Gemini**, creada en su propia cuenta o proyecto de Google, para no compartir cuota.

## 2. Cómo trabajar cada día

```bash
git switch main && git pull                  # empezar siempre desde main actualizado
git switch -c feat/wpN-lo-que-hago           # una rama por tarea
# ... trabajar, probar ...
git add <tus archivos> && git commit -m "feat(wpN): qué has hecho"
git push -u origin feat/wpN-lo-que-hago      # y abrir un PR hacia main en GitHub
```

- **PR hacia `manueladominguez/daiist-group-project` · `main`**, nunca hacia el repositorio del profesor (`ami232`).
- **Otra persona revisa y aprueba** el PR en GitHub (botón *Review changes → Approve*) antes de fusionarlo. Queda registrado y cuenta como contribución.
- **Cada fusión en `main` publica la app en Render.** No se fusiona nada que no arranque con `uv run python app.py`.
- Cada PR añade su línea en `CONTRIBUTIONS.md`, con el nombre de quien hizo el trabajo y la ayuda de IA si la hubo.

## 3. Reglas comunes

1. **Contratos:** `maintai/schemas.py` define qué se pasan los módulos. No se cambia dentro de otro trabajo: se hace en un PR propio y se avisa al equipo.
2. **Cada archivo tiene un dueño** (tabla de la sección 4). Para tocar el archivo de otro, se habla antes con esa persona.
3. **Dependencias:** solo se añaden con `uv add`, avisando antes, y subiendo `pyproject.toml` y `uv.lock` juntos.
4. **El test no se toca** para elegir modelos, umbrales, prompts ni el umbral de drift. Solo train y validación. El test se usa una vez, para el informe.
5. **Hechos, simulaciones y supuestos** se etiquetan siempre por separado.
6. **Gemini:** cada llamada gasta cuota (500 al día con Flash-Lite y 20 con 3.8 Flash). Hay que usar la caché y evitar repetir pruebas.
7. **Nada de claves ni datos personales** en el repositorio, los logs o los ejemplos.
8. **Registros de prueba:** todo lo que tenga versiones `stub-` (los smoke tests del 25 nov) se excluye de la evaluación y del bandit.

## 4. Paquetes de trabajo

| WP | Qué entrega | Archivos de los que es dueño | Depende de | Puede empezar |
|---|---|---|---|---|
| **WP1 · Componente A** | Datos AI4I con licencia, splits fijos, comparación de modelos en validación, calibración, umbral con el modelo económico completo, `model_card.json`, `component_a.assess()` real | `data/`, `training/`, `models/`, `maintai/component_a.py` | — | **Ya** |
| **WP2 · Herramientas y economía** | Datos simulados de repuestos, calendario y políticas; herramientas completas; script de análisis de sensibilidad | `maintai/tools.py`, `maintai/data/`, `maintai/economics.py` | — | **Ya** |
| **WP3 · Agente** | Cliente Gemini (límite por minuto, presupuesto, 429, tokens, caché), agente con ≤ 3 llamadas, prompts versionados, guardrails, modo degradado | `maintai/llm.py`, `maintai/agent.py`, `maintai/prompts/`, `maintai/guardrails.py`, `pipeline.py` | — (usa los stubs de WP1 y WP2) | **Ya** |
| **WP4 · Evaluación** | `eval/cases.csv` (objetivo 90), baselines B1/B2, `evaluate.py`, juez y rúbrica, etiquetas humanas | `eval/cases.csv`, `eval/human_labels.csv`, `evaluate.py`, `eval/judge.py`, `eval/baselines.py` | Splits de WP1 para los casos de test | **Ya** los casos `hard`, `review` y `out_of_scope` escritos a mano |
| **WP5 · Operación** | Lista de features de drift, `reference.csv`, `shifted.csv`, calibración del umbral de drift en validación, alertas, panel de cuota | `maintai/features.py`, `eval/reference.csv`, `eval/shifted.csv`, `monitoring/`, `monitor.py`, `eval/drift_calibration.py` | Splits de WP1 | **Ya** el diseño; datos tras WP1 |
| **WP6 · Bandit (bonus)** | Thompson sampling desde los logs de Postgres, prior simulado (máx. 10 pseudo-observaciones por brazo), panel de RL, campaña de feedback | `maintai/bandit.py`, `eval/simulate_bandit.py`, `maintai/rl_monitor.py` | Estrategias del agente (WP3) para la parte live | **Ya** el bandit y el simulador |
| **WP7 · Interfaz** | Interfaz SaaS: logo, favicon, CSS, formulario, tarjetas de riesgo y orden, traza del agente, dashboards | `app.py`, `assets/` | Contratos (ya están) | **Ya**, contra los stubs |
| **WP8 · Informe y demo** | `REPORT.md`, decisión, presentación, vídeo de respaldo | `REPORT.md`, `docs/` | Todos | Ya el esqueleto; cifras al final |

**Calendario:** M1 jueves por la noche (flujo completo con A y agente reales en local) · M2 viernes (en Render, `check.py` en verde) · M3 sábado por la mañana (bandit live y feedback real) · M4 domingo 22:00 (congelación) · entrega lunes.

## 5. Reparto propuesto según el tamaño del equipo

| Personas | Reparto |
|---|---|
| 3 | A: WP1 + WP5 · B: WP2 + WP3 · C: WP4 + WP6 + WP7 · WP8 entre todos |
| 4 | A: WP1 + WP5 · B: WP2 + WP3 · C: WP4 + WP8 · D: WP6 + WP7 |
| 5 | A: WP1 · B: WP3 · C: WP2 + WP4 · D: WP5 + WP6 · E: WP7 + WP8 |
| 6 | A: WP1 · B: WP3 · C: WP4 · D: WP5 · E: WP6 · F: WP2 + WP7 · WP8 coordinado por una persona |

Aunque cada uno tenga su parte, **todos tienen que poder explicar todo** en las preguntas de la presentación.

## 6. Brief de diseño para WP7 (elección final dentro de WP7)

Objetivo: aspecto de SaaS industrial excepcionalmente profesional, con identidad corporativa propia (logotipo original y favicon), CSS personalizado, visualizaciones avanzadas e indicadores de negocio y financieros **calculados con datos reales del sistema** (logs y evaluación, nunca cifras inventadas). Fondo definitivo **Cloud Grey `#F1F5F9`** y tarjetas **blancas `#FFFFFF`**. Dos paletas finalistas; los colores se ajustan donde haga falta para cumplir la accesibilidad.

| Paleta | Primario oscuro | Acento | Claro | Gris |
|---|---|---|---|---|
| Aerospace Blue | `#102E4A` | `#007C9E` | `#A8D8EA` | `#718096` |
| Emerald Systems | `#103B35` | `#008F70` | `#A9D5C6` | `#64748B` |

Contraste medido con la fórmula WCAG 2.1. Para texto normal, AA exige 4,5:1.

| Color | Sobre blanco | Sobre `#F1F5F9` | Uso recomendado |
|---|---|---|---|
| `#102E4A` / `#103B35` | 13,9 / 12,4 | 12,7 / 11,3 | Texto principal, cabecera, botones con texto blanco |
| `#007C9E` (Aerospace) | 4,80 ✅ | 4,38 ⚠️ | Enlaces y botones sobre tarjetas blancas; sobre el fondo gris, solo texto grande (≥ 18,66 px en negrita o ≥ 24 px) |
| `#008F70` (Emerald) | 4,07 ⚠️ | 3,71 ⚠️ | Solo texto grande, iconos o rellenos; para texto normal habría que oscurecerlo |
| `#A8D8EA` / `#A9D5C6` | 1,5 | 1,4 | Solo fondos, rellenos y gráficos, **nunca texto** |
| `#718096` (gris Aerospace) | 4,02 ⚠️ | 3,67 ⚠️ | Etiquetas grandes; para texto pequeño usar uno más oscuro |
| `#64748B` (gris Emerald) | 4,76 ✅ | 4,34 ⚠️ | Texto secundario sobre tarjetas blancas |

Restricciones técnicas para WP7:
- Mantener `serve()` y un único campo de texto para el endpoint `/run`: `check.py` envía un solo argumento. El formulario visible construye ese JSON.
- Recursos de Render: 512 MB de memoria y 0,15 CPU. El consumo real **todavía no está medido**: las gráficas de *Metrics* no mostraban datos el 25 nov. Se mide cuando entren los modelos definitivos (WP1 y WP3) y después de cada dependencia nueva. Preferir los gráficos nativos de Gradio antes que añadir librerías pesadas.
- Colores semánticos (bajo / medio / alto / crítico, o revisión) independientes del acento de marca.
- Logo y favicon originales en SVG dentro de `assets/`. Nada que imite marcas existentes.
- Hallazgo de la plantilla: en Gradio 6.31, los handlers con `api_name=False` aparecen publicados como endpoints llamados `false`, `false_1`... (lo hemos visto en `/config` de la app online). Hay que comprobar en la documentación de Gradio cuál es la forma actual de ocultarlos.
