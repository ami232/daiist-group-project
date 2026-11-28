# MaintAI · Plan de trabajo del equipo

Guía interna para trabajar en paralelo sin pisarnos. El plan completo está en el master plan; aquí solo lo que cada persona necesita para empezar. Las decisiones y supuestos del business case están en `REPORT.md` (aprobados por Manuela, pendientes de confirmación del resto del equipo).

Entrega: **lunes 30 nov, 23:59**. App: https://maintai-238a.onrender.com

## 0. Estado real del proyecto (actualizado el sábado 28 nov, 10:45)

Fase actual: **documentación, integración y validación**. Ya no se añaden funcionalidades nuevas: solo lo necesario para cumplir los requisitos del profesor y conseguir evidencia sólida. El orden de integración, las pruebas tras cada merge y la checklist de entrega están en [`docs/INTEGRATION_CHECKLIST.md`](INTEGRATION_CHECKLIST.md).

### 0.1 Estado por paquete (`main` = commit `2e9893f`)

| WP | Responsable | Estado | Qué hay | Qué falta |
|---|---|---|---|---|
| WP0 · Base | Manuela | ✅ En `main` | Contratos (`schemas.py`), `render.yaml`, Render + PostgreSQL | — |
| WP1 · Componente A | Ignacio (GitHub `94921234`) | ✅ En `main` (PR #9) y **desplegado en Render** | Gradient Boosting con features derivadas y calibración sigmoide (`maintai-gb-sigmoid-v1`), umbral 0,19 elegido con el modelo económico **en validación**, banda de incertidumbre, 4 clasificadores de modo de fallo, `models/model_card.json`. En validación: recall 0,81, precisión 0,93, PR-AUC 0,883. **El test aún no se ha usado** (`test_evaluated: false`) | Evaluación final en test (una sola vez, la hace WP4). El docstring de `component_a.py` todavía dice "STUB" (avisar a Ignacio). Su rama tiene 3 commits más sin PR, con unas 2.400 palabras para el REPORT (ver 0.5) |
| WP2 · Herramientas y economía | Marta (`martabermejo`) | ✅ En `main` (PR #8) | `maintai/data/*.json`, herramientas completas, modelo de costes con bootstrap, `eval/economics_sensitivity.py`, 26 tests (14 + 12) | Usar las herramientas desde el agente real (WP3) |
| WP3 · Agente | Jacobo | ✅ En `main` (PR #12) y **desplegado en Render** | Cliente Gemini (`gemini-3.5-flash-lite`), agente `agent-v1` con 1 llamada en riesgo bajo y 3 en el resto, decisión sin estilo y guardrails que rechazan diferencias, presupuesto diario reconstruido desde los logs, límite por minuto, reintentos, modo degradado, caché (`off` en Render), 26 tests con Gemini simulado | Validación con Gemini real del equipo (invariancia, coste, latencia, memoria en Render); caché de la evaluación (`eval/llm_cache/`) tras ejecutar WP4 |
| WP4 · Evaluación | Ana Maria (rama `feat/wp4-evaluation`) | 🟡 Terminado en su rama `feat/wp4-evaluation`, **sin PR todavía** | 41 casos (16 normal, 10 hard, 8 review, 7 out_of_scope), baselines B1/B2 ajustados con train + validación, juez LLM con rúbrica de 4 puntos y caché, herramienta de etiquetado, `evaluate.py` completo, 8 tests | Abrir el PR; `eval/human_labels.csv` sigue vacío (el profesor y check.py exigen al menos 20 etiquetas propias para medir el acuerdo del juez; nuestro objetivo interno es 30); ejecutar la evaluación **cuando el agente real esté**, porque usa el test (una vez) y gasta cuota del juez |
| WP5 · Operación | Por confirmar | 🔴 No empezado | `monitor.py` (PSI, alertas) de la plantilla | `eval/reference.csv` y `eval/shifted.csv` (**check.py se para en el paso 2 por esto**), `main.py reference` → `monitoring/reference_features.csv` (paso 4), umbral de drift |
| WP6 · Bandit | Manuela | ✅ En `main` y desplegado (sin evidencia real) | Bandit, simulador, calibración, métricas de feedback, límites por sesión, exportación, 41 tests; escenarios de la campaña ya **definitivos** con el modelo real | Campaña real, cuando WP3 esté desplegado: [`docs/RL_FEEDBACK_PROTOCOL.md`](RL_FEEDBACK_PROTOCOL.md) |
| WP7 · Interfaz | Manuela | ✅ En `main` | App y Monitoring, accesible, 375 px, `/run` único endpoint; compatible con WP1, WP2 y WP3 (comprobado con Gemini simulado) | Correcciones de integración en `fix/integration-polish` (umbral frente a break-even, modos de fallo independientes, etiqueta `part_out_of_stock`) |
| WP8 · Informe | Manuela + todos | 🟡 En curso | Business case y RL resumidos | Ver presupuesto en 0.5 |

*"Por confirmar": todavía no hay commits de ese WP. Cada persona confirma su WP en el chat del equipo y añade su fila en `CONTRIBUTIONS.md` (Ignacio y Ana Maria aún no tienen fila de miembro; check.py avisa de filas vacías).*

### 0.2 Render y PostgreSQL (comprobado el 27–28 nov)

- La app responde (HTTP 200) y `/run` sigue siendo el único endpoint, con el parámetro `raw_input`.
- Render sirve ya el modelo real (WP1) y el **agente real (WP3)**: el banner de stubs ha desaparecido. **Desde ahora, cada petición online puede gastar cuota de Gemini.**
- **Persistencia en PostgreSQL verificada** (exportación del 27 nov, 21:24, 4 peticiones): la petición hecha desde la App tras el hardening guarda `session_tag` (16 caracteres), `bandit_policy_from = logs`, los `arm_probs` de `ts-v1` y las versiones reales. No es elegible para RL (agente stub en ese momento, orden degradada y sin valorar), como se esperaba. Detalle en `INTEGRATION_CHECKLIST.md`, secciones 6.4 y 6.5.
- **Pendiente (Manuela, manual):** memoria y CPU en Render Metrics con el modelo y el agente reales (estimación local: unos 315 MB de pico).

### 0.3 check.py hoy

| Paso | Estado en `main` | Con WP4 fusionado | Bloqueado por |
|---|---|---|---|
| 1 · Variables de entorno | ✅ | ✅ | — |
| 2 · Archivos de evaluación | ❌ `cases.csv` vacío | ⚠️ `cases.csv` pasa (41), falla `reference.csv` (0 de 30) | WP4 (PR) y WP5 (`reference.csv`, `shifted.csv`) |
| 3 · App y logs | No se llega (check.py se para en el 2) | ✅ Simulado aparte (27 nov): registro válido con todos los campos y 9 features; en local, sin caché, el agente pasa a modo degradado sin llamar a Gemini | — |
| 4 · Drift | No se llega | ❌ Falta `monitoring/reference_features.csv` | WP5 (después de WP1, porque `p_fail` es una feature de drift) |
| 5 · Repositorio | No se llega | Avisos: filas de miembros vacías; `human_labels.csv` vacío | Todos (CONTRIBUTIONS), WP4 (etiquetas) |
| 6 · `--url` (Render) | Sin probar hoy | — | Se hace al final; escribe una petición real en PostgreSQL |

### 0.4 Dependencias y tareas pendientes

```
WP1 ✅ ──┬──► WP5 (referencia de drift)          ──► check.py pasos 2 y 4
WP2 ✅ ──┤
         └──► WP3 ✅ (agente real) ┬──► WP4: evaluación end-to-end y test (una vez) ──► Evaluation, Failures, Decision
                                   └──► WP6: campaña de feedback (prompts congelados) ──► bonus
WP4 (PR) ── ya se puede fusionar: no depende del agente para pasar sus tests
```

| # | Tarea | Responsable | Límite propuesto | Bloquea |
|---|---|---|---|---|
| 1 | ~~Agente real~~ ✅ fusionado (PR #12). Queda: validación con Gemini real y caché de la evaluación | WP3 | Sáb 28, 14:00 | Evaluación, campaña |
| 2 | `eval/reference.csv`, `eval/shifted.csv`, `main.py reference`, umbral de drift | WP5 | Sáb 28, 14:00 | check.py pasos 2 y 4 |
| 3 | Abrir el PR de WP4 (resolver el conflicto trivial de `CONTRIBUTIONS.md`) y etiquetar órdenes en `human_labels.csv` (mínimo obligatorio 20, objetivo interno 30) | WP4 | Sáb 28, 12:00 | Acuerdo del juez |
| 4 | PR con el texto de WP1 para el REPORT, **recortado** (ver 0.5) | WP1 | Sáb 28, 18:00 | Informe |
| 5 | Ejecutar `evaluate.py` con el sistema final (test una vez, cuota del juez autorizada) | WP4 | Dom 29, 12:00 | Evaluation, Decision |
| 6 | Campaña de feedback real (150 valoraciones si hay cuota; puede no alcanzarse) | Manuela + todos | Dom 29, 18:00 | Bonus |
| 7 | Secciones del REPORT de cada WP | Cada responsable | Dom 29, 20:00 | Entrega |
| 8 | Congelación, check.py en un clon limpio, tag `final`, PR al profesor, Blackboard | Manuela | Dom 29, 22:00 / lun 30 | Entrega |

### 0.5 Presupuesto de palabras del REPORT (unas 3.000 sin tablas)

**Aviso:** el texto que WP1 tiene en su rama (sin PR) se fusiona sin conflictos con el nuestro, pero lleva el total a **unas 4.360 palabras**, y todavía faltan WP3, WP4 y WP5. Cada sección debe ajustarse a su presupuesto; los detalles van a `docs/` (como ya hemos hecho con el business case y el RL).

| Sección | En `main` | Rama de WP1 | Objetivo | Quién |
|---|---|---|---|---|
| Cabecera | 52 | — | 52 | — |
| Business case (con los criterios preregistrados, 420 palabras, **sin cambios** desde `1d0135f`) | 887 | — | 887 | Manuela |
| Architecture | plantilla | 307 | 250 | WP1 + WP3 |
| Data | plantilla | 474 | 150 | WP1 + WP2 |
| Evaluation (componentes, end to end, coste y latencia) | plantilla | 1.257 | 450 | WP4 (+ WP1, WP2) |
| Operations | 180 | — | 300 | WP5 + Manuela |
| Failures | plantilla | 137 | 150 | WP4 |
| Decision | plantilla | 287 | 150 | Todos |
| Bonus: reinforcement learning | 479 | — | 450 | Manuela |
| How to run it | plantilla | 183 | 60 | Manuela |
| AI use | 117 | — | 120 | Todos |
| **Total** | **unas 1.930** | **unas 4.360** | **unas 3.020** | |

Los textos largos que se han quitado del REPORT están completos en [`docs/BUSINESS_CASE.md`](BUSINESS_CASE.md) y en el apéndice de [`docs/WP6_BANDIT.md`](WP6_BANDIT.md).

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

**Calendario (revisado el 27 nov, noche):** WP1 y WP2 ya están en `main`; WP4 está terminado en su rama. Sábado 28 (WP3 y WP5 integrados y en Render, PR de WP4) · domingo 29 por la mañana (evaluación final) · domingo 29 hasta las 18:00 (campaña de feedback, si hay cuota) · domingo 22:00 (congelación) · lunes 30 (check.py en un clon limpio, tag `final`, PR y Blackboard). Detalle en la sección 0.4.

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
