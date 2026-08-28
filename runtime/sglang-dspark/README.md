# Port del dial rank-1 a SGLang (Qwen3.8-27B, motor DSpark)

Hermano del port de vLLM de `../vllm-0.27.1/`. Mismo fichero de direcciones, mismo
significado de lambda, otro motor: la receta publica
[`MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark),
en su modo **DSpark** (`start-dspark.sh`), que es el default de ese repo.

La imagen de la receta **no trae el hook**: cambiar de motor a secas borra el
`-uncensored` del residente. Esta capa lo devuelve.

> **NO MEDIDO EN GPU TODAVIA.** Lo que esta verificado es: (a) las doce anclas existen
> exactamente una vez en el arbol de la imagen real —extraido del registry, no supuesto—
> y los cinco ficheros resultantes parsean; (b) el algebra, en CPU
> (`test_projection_equivalence.py`). Lo que falta es la unica prueba que importa de
> verdad: que con lambda=0 la salida sea identica a la del pod viejo, y que con lambda=1
> el refusal caiga. Eso solo se ve sirviendo.

## Que se conserva del port de vLLM

- **El fichero `hf/refusal_dirs_qwen38.safetensors`.** 128 direcciones — 48
  `linear_attn.out_proj`, 16 `self_attn.o_proj`, 64 `mlp.down_proj`, capas 0..63, hidden
  5120 — mas un **`coef` por modulo**. Heretic no abla con la misma fuerza en cada capa,
  asi que un lambda uniforme sin `coef` no reproduce el perfil. `lam=1` = el perfil de
  `Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored`; `lam=0` = base bit-exacto.
- **Fail-closed.** Al terminar de construir el modelo se comprueba que las 128
  direcciones han sido reclamadas. Un modelo a medio ablar no da error y solo se nota
  midiendo el refusal rate; preferimos no arrancar.
- **Lambda como tensor en device, mutado in-place.** SGLang captura CUDA graphs: un
  `float` de Python se hornea en la captura y el decode se queda con el valor de entonces
  para siempre, sin un aviso.
- **La superficie HTTP.** `/admin/refusal_lambda`, GET y POST `{"lambda": x}`, con el
  mismo cuerpo de respuesta (`lambda`, `consistent`, `per_rank`). No es cosmetica:
  LiteLLM lleva ese `admin_url` cableado y el panel DGX lee y conmuta el dial por ahi.

## Que cambia, y por que

| | vLLM 0.27.1 | SGLang (esta capa) |
|---|---|---|
| donde se aplica | hook por modulo (`RefusalProjection`) | dentro del forward, por anclas |
| lambda por peticion | si (`cache_salt: refusal:<x>`) | **no** |
| dial global | `/admin/refusal_lambda` | igual, envolviendo `/set_internal_state` |
| drafter | MTP del propio checkpoint, sin ablar (`MTP_MODE=off`) | DSpark, otro checkpoint, sin ablar |

**El lambda por peticion no se porta.** En vLLM viajaba en `cache_salt` y costo cablear
el Model Runner V2; SGLang no tiene plumbing equivalente y no se inventa aqui. Decision
del owner (28-08-2026). Consecuencia concreta, porque muerde: la entrada
`qwen38-27b-uncensored` de LiteLLM sella `extra_body: {cache_salt: "refusal:1.0"}`, y
con este motor **ese sello no hace nada**. El alias uncensored responde con el lambda
GLOBAL que tenga el pod en ese momento — que arranca en 0, o sea censurado. Para que
signifique algo hay que subir el dial (panel o `POST /admin/refusal_lambda`), y entonces
lo sube para TODO el pod, incluido el alias normal. Un pod, un lambda.

**Los `--forward-hooks` nativos de SGLang no valen.** Se registran DESPUES de capturar
los grafos (`model_executor/model_runner_components/cuda_graph_setup.py` lo dice en un
comentario: *"capture stays hook-free and hooks fire only on the eager forward path"*).
Con cuda graph en decode, un hook nativo abla el prefill y **no** abla el decode:
ablacion parcial y muda. Por eso el parche va dentro del forward del modulo.

**El drafter DSpark no va ablado.** Es otro checkpoint y otra arquitectura
(`models/dspark.py`), y no tenemos direcciones para el. Con lambda>0 target y drafter
dejan de estar alineados y la acceptance baja (~20% medido en el lado vLLM con MTP en
los temas en que se enciende el dial). La **salida** no cambia: el rejection sampling la
fija a la del target. Es coste de velocidad, no de correccion.

## Los doce sitios

`patch_sglang_qwen38_27b.py`, fail-closed por anclas — si una version futura de la
imagen mueve cualquiera de ellos, el build muere:

| # | fichero | sitio |
|---|---|---|
| S0 | `models/qwen3_5.py` | import del payload |
| S1 | | `init_from_env` ANTES de construir las capas |
| S2 | | `verify_all_consumed()` tras `make_layers` (no en `is_nextn`) |
| S3/S4 | | `Qwen3_5GatedDeltaNet`: resolver + proyectar `out_proj` (48) |
| S5/S6 | | atencion completa: resolver + proyectar `o_proj` (16) |
| M0 | `models/qwen2_moe.py` | import del payload |
| M1/M2 | | `Qwen2MoeMLP`: resolver + proyectar `down_proj` (64), **las dos ramas** |
| D0–D2 | `managers/scheduler.py` | `refusal_lambda` como comando de worker en `set_internal_state` |
| D3 | | el lambda vivo en `get_internal_state` (el readback del panel) |
| H0 | `entrypoints/http_server.py` | `/admin/refusal_lambda` GET + POST |

Dos detalles que parecen menores y no lo son:

1. **`Qwen2MoeMLP.forward` tiene DOS returns.** La rama fusionada
   (`_enable_silu_fp4_quant_fusion`, el SiLU+mul+FP4-quant de FlashInfer) sale por su
   propio `return` y es justo la que se enciende con el checkpoint NVFP4. Parchear solo
   la de abajo dejaria el modelo **sin ablar en produccion y ablado en las pruebas**.
2. **El naming.** Las claves del fichero son del checkpoint
   (`model.language_model.layers.N.mlp.down_proj`) y SGLang mete otros niveles por
   encima. `resolve()` extrae el indice de capa y los dos ultimos componentes del camino,
   asi que da igual cuantos prefijos anada el motor. Y **corta** los prefijos del drafter
   (`mtp`, `draft`, `dspark`, `dflash`, `nextn`, `eagle`), que reindexan desde 0 y
   reclamarian direcciones del backbone que no son suyas.

## Construir

```sh
# desde la raiz del repo; el contexto es la raiz porque el Dockerfile copia hf/
docker buildx build --platform linux/arm64 \
  -t harbor.lan.e-dani.com/homelab/sglang-qwen38-27b-rank1-arm64:v0.1.0 \
  -f runtime/sglang-dspark/Dockerfile .
```

Capa fina: no clona sglang, no compila kernels, no necesita GPU. Los tres gates del
build son (1) las doce anclas, (2) que los cinco ficheros parseen y las dos rutas admin
existan, (3) que el fichero de direcciones sea el que este hook espera (128 modulos,
hidden 5120, unitarias, con `__coefs__`).

El `--site` se **resuelve** en el build (`importlib.util.find_spec`), no se cablea:
parchear un arbol que el interprete no importa no da error y no hace nada.

## Comprobar

```sh
python3 runtime/sglang-dspark/test_projection_equivalence.py   # CPU, sin motor
```

```
1) out_proj   peso editado == proyeccion   err_max=1.506e-06
1) o_proj     peso editado == proyeccion   err_max=2.713e-06
1) down_proj  peso editado == proyeccion   err_max=1.884e-06
2) lambda=0 identidad bit a bit  OK
3) set_lambda in-place  OK  (lam=1.50, mismo tensor)
4) forma incorrecta -> RuntimeError  OK
8) doble reclamacion -> RuntimeError  OK
```

La 8 cierra el unico agujero que dejaba el gate de arriba: `verify_all_consumed()`
contaba con un `set`, que ve la FALTA pero no el EXCESO. Si dos modulos reclamasen la
misma direccion —otro subarbol cuyo nombre tambien case con `layers.N.<a>.<b>`— esa capa
quedaria proyectada a 2*lambda y el conteo seguiria diciendo 128/128. Ahora es un
contador y se exige exactamente una vez.

## Usar

```sh
curl -s localhost:8888/admin/refusal_lambda
curl -s -XPOST localhost:8888/admin/refusal_lambda -H 'Content-Type: application/json' \
     -d '{"lambda": 1}'
# el canal interno, equivalente:
curl -s -XPOST localhost:8888/set_internal_state -H 'Content-Type: application/json' \
     -d '{"server_args": {"refusal_lambda": 1}}'
```

Rollback sin rebuild: quitar `SGLANG_REFUSAL_DIRS` y reiniciar. Sin esa variable el hook
ni se construye y la imagen se comporta como la publica de la receta.

**No exponer `/admin/*` por el ingress publico.** Esto no lo puede garantizar el motor:
es una ruta HTTP mas, y quien decide que sale a internet es el ingress / AgentGateway.

## Orden de pruebas tras el rollout — dos cambios, dos pasos

El rollout no es solo "el dial": cambia el MOTOR entero (vLLM -> SGLang + DSpark). Si se
mide todo a la vez y algo sale raro hay dos causas candidatas y ninguna es atribuible.

1. Arrancar con `SGLANG_REFUSAL_LAMBDA=0` y comparar una generacion greedy contra la del
   pod viejo con el mismo prompt. Si difieren **con lambda=0**, algo del parche cambio la
   numerica y no es el lambda.
2. Solo entonces subir el dial a 1 y re-medir refusal **y** accept-len.
3. El barrido de lambda del port de vLLM (1.0 es el punto de operacion; 1.5 quita los
   ultimos disclaimers pero degrada calidad; 2.5 no es "mas uncensored" sino que el
   refusal vuelve a la base) esta medido en **aquel** motor. Es rango de partida, no
   calibracion: hay que re-barrer.
