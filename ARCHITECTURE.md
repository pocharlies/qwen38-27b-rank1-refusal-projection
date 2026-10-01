# ARCHITECTURE.md — qwen38-27b-rank1-refusal-projection

Experimento publicado: el mismo dial de proyección de rechazo de rango 1 en runtime (λ en caliente, vectores de 2,6 MB, checkpoint base intacto) aplicado a Qwen3.8-27B, con dos runtimes: vLLM 0.27.1 (MTP k=3) y SGLang + DSpark (`runtime/sglang-dspark/`). Tronco: **`main`**. **Estado: histórico**: el comentario del manifiesto de LiteLLM (`k8s-litellm-pocharlies/k8s/manifest.yaml`, ~línea 420) registra `qwen38-27b` y `qwen38-27b-uncensored` como RETIRADOS el 26-09-2026 (el despliegue `vllm-qwen38-27b-uncensored` figura a 0 réplicas).

## Clientes y versiones
- Sin clientes de producto. Artefactos: `vllm/` (parches de gpu_worker, model_runner, scheduler y router `/admin/refusal_lambda`), `runtime/vllm-0.27.1/` y `runtime/sglang-dspark/` (Dockerfile + `patch_sglang_qwen38_27b.py`, 14 anclas: si una se mueve el build falla), `recipes/*.yaml` (recetas SparkRun), `tools/`, `bench/` y `deploy/`.
- Medido en un GB10 compartido por varias cargas (por eso `gpu_memory_utilization 0.35`): las cifras absolutas (~20 tok/s) no son la velocidad del modelo en un Spark dedicado; benchmarks fechados 2026-08-17/22/29 en `hf/benchmarks/`, no son el estado actual.

## Dependencias (ambos sentidos)
- **De**: checkpoint Qwen3.8-27B NVFP4 (Unsloth) y MTP nativo; receta pública `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark` para el runtime DSpark; vLLM 0.25.2/0.27.1; SGLang.
- **Dónde viven los datos**: vectores y ficha en Hugging Face (`pocharlies/qwen38-27b-uncensored-abliterated-refusal-directions`, subidos desde `hf/`); la imagen vLLM de la receta se referenciaba como `vllm-qwen38-rank1` en Harbor (`recipes/qwen38-27b-nvfp4-refusal-dial.yaml`); los pesos, en el almacén de pesos del clúster, no en este repo.
- **Quién depende**: nadie en código; LiteLLM conserva el alias retirado.

## Stack
Python (parches y herramientas), vLLM, SGLang, CUDA/PyTorch, Dockerfiles y Jobs de Kubernetes. No se usa: pesos modificados; el dial va por la clave `cache_salt: "refusal:<x>"` por petición (el sello aísla la caché de prefijos entre λ distintos).

## Componentes compartidos (canónicos)
Mecanismo gemelo de `deepseek-v4-flash-rank1-refusal-projection`; el vigente para el residente actual está en `k8s-ai-pocharlies/k8s/qwen38-flash-next-ursucipian-mod/refusal/`. No añadir otro fork.

## Cómo se construye aquí
No es servicio. Reproducir con `tools/extract_refusal_dirs_qwen.py` → `tools/verify_qwen38_rank1.py` → `bench/comprehensive/compare_full.py`; imágenes con `deploy/job-build-rank1g.yaml` y los Dockerfiles de `runtime/`. En SGLang los `--forward-hooks` nativos NO sirven (se registran tras capturar los CUDA graphs: ablacionan prefill y no decode): el parche va dentro del forward del módulo.

## Tests y validaciones
`tools/test_recipe.py`, `tools/test_per_request.py`, `tools/negctl_per_request.py`, `runtime/sglang-dspark/test_projection_equivalence.py`; resultados en `bench/results/` y `hf/benchmarks/`. Sin CI.

## CI/CD y despliegue
Ninguno. **Antes de lanzar un Job o carga en los Sparks se lee el ConfigMap `gpu-arbiter-state` (ns `comfyui`)**: con `llm-tp` efectivo no cabe otra carga y con `phase: switching` no se toca nada; los Jobs de build de `deploy/` no se lanzan sin esa lectura.

## Decisiones y trampas
- Resultado (StrongREJECT, n=60/arm; benchmark fechado 2026-08-17): λ=0 → 32/60 rechazos estrictos; **λ=1 → 0/60**, GSM8K 84→81 % y MMLU-Pro 76,8→75,0 % sin diferencia detectable; λ=1,5 degrada la calidad; λ=2,5 devuelve el rechazo y la generación se vuelve patológicamente lenta (efecto no monótono).
- Candidato a archivar (propuesta, C5): modelo retirado y mecanismo duplicado; vectores en Hugging Face.
