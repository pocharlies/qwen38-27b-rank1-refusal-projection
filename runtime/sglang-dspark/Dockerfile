# sglang-qwen38-27b-rank1-arm64:v0.1.0
#   = lmsysorg/sglang:qwen38-27b (la imagen de la receta publica) + proyeccion rank-1.
#
# UNA sola capa fina encima de la imagen que publica
# `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`. No clona sglang, no compila kernels, no
# necesita GPU: copia el payload, parchea diez anclas y comprueba.
#
# Construir desde ESTE directorio:
#   docker buildx build --platform linux/arm64 \
#     -t harbor.lan.e-dani.com/homelab/sglang-qwen38-27b-rank1-arm64:v0.1.0 \
#     -f Dockerfile ../..            # el contexto es la RAIZ del repo (por hf/)
#
# El parche es FAIL-CLOSED por anclas: si una version futura de la imagen mueve
# cualquiera de los diez sitios, el build MUERE aqui en vez de producir una imagen a
# medio parchear — que es indistinguible de una sana mirando una respuesta.
ARG BASE=docker.io/lmsysorg/sglang:qwen38-27b@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6
FROM ${BASE}

COPY runtime/sglang-dspark/payload/ /opt/refusal/payload/
COPY runtime/sglang-dspark/patch_sglang_qwen38_27b.py /opt/refusal/
COPY runtime/sglang-dspark/test_projection_equivalence.py /opt/refusal/
COPY hf/refusal_dirs_qwen38.safetensors /opt/refusal/refusal_dirs_qwen38.safetensors

# El `--site` se RESUELVE, no se cablea: la imagen instala sglang desde
# /sgl-workspace/sglang, pero lo que importa es el paquete que el interprete importa de
# verdad. Cablear la ruta es como parchear un arbol que nadie carga: cero error, cero
# efecto.
RUN SITE="$(python3 -c 'import importlib.util as u; print(u.find_spec("sglang").submodule_search_locations[0])')" \
    && echo "sglang en ${SITE}" \
    && python3 /opt/refusal/patch_sglang_qwen38_27b.py --site "${SITE}" --payload /opt/refusal/payload

# Gate 1: los cuatro ficheros tocados siguen siendo Python valido, y el payload esta
# donde el parche lo importa (`from sglang.srt import refusal_projection`).
RUN python3 - <<'PY'
import ast, importlib.util, pathlib
site = pathlib.Path(importlib.util.find_spec("sglang").submodule_search_locations[0])
for f in ("srt/models/qwen3_5.py", "srt/models/qwen2_moe.py",
          "srt/managers/scheduler.py", "srt/entrypoints/http_server.py",
          "srt/refusal_projection.py"):
    ast.parse((site / f).read_text())
    print("sintaxis OK", f)
# La superficie que consumen LiteLLM y el panel: si un dia desaparece, que se
# entere el build y no el operador buscando por que el dial no responde.
http = (site / "srt/entrypoints/http_server.py").read_text()
assert http.count('"/admin/refusal_lambda"') == 2, "faltan las rutas GET/POST del dial"
print("rutas /admin/refusal_lambda OK")
PY

# Gate 2: el fichero de direcciones es el que este hook espera — 128 modulos, hidden
# 5120, todas unitarias y con `__coefs__`. Sin torch a proposito: importarlo aqui pide
# 2-3 GB en un nodo que vive al limite.
RUN python3 - <<'PY'
import json, struct, math
p = "/opt/refusal/refusal_dirs_qwen38.safetensors"
with open(p, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(n))
    blob = fh.read()
keys = [k for k in header if not k.startswith("__")]
assert len(keys) == 128, len(keys)
assert "__coefs__" in header, "falta __coefs__: fichero de la generacion anterior"
order = json.loads(header["__metadata__"]["coef_order"])
assert set(order) == set(keys), "coef_order no casa con las direcciones"
import array
for k in keys:
    m = header[k]
    assert m["dtype"] == "F32" and m["shape"] == [5120], (k, m["dtype"], m["shape"])
    a, b = m["data_offsets"]
    v = array.array("f"); v.frombytes(blob[a:b])
    nrm = math.sqrt(sum(x * x for x in v))
    assert 0.99 <= nrm <= 1.01, (k, nrm)
print(f"direcciones OK: {len(keys)} modulos, hidden 5120, todas unitarias")
PY

# Por defecto DESACTIVADA: sin SGLANG_REFUSAL_DIRS el hook ni se construye y la imagen
# se comporta EXACTAMENTE como su base (ese es el rollback sin rebuild). LAMBDA=0
# arranca con el dial a cero — el estado uncensored se ENCIENDE, no es el default.
# Las dos van en el manifiesto, no aqui, para que el dial se vea en GitOps.
ENV SGLANG_REFUSAL_LAMBDA=0

LABEL org.opencontainers.image.title="sglang-qwen38-27b-rank1-arm64" \
      org.opencontainers.image.version="v0.1.0" \
      org.opencontainers.image.base.name="docker.io/lmsysorg/sglang:qwen38-27b" \
      org.opencontainers.image.source="https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection" \
      ai.e-dani.recipe="https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark" \
      ai.e-dani.model="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead" \
      ai.e-dani.rank1-refusal="target; drafter DSpark SIN ablar; lambda global por /set_internal_state" \
      ai.e-dani.arch="linux/arm64 (sm121)"
