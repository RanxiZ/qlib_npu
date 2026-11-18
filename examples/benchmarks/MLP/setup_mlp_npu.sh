#!/usr/bin/env bash
# Run once from any directory; activate the printed environment before qrun.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}
if [ ! -f "$CANN_ENV" ]; then echo "Set CANN_ENV to the installed CANN set_env.sh" >&2; exit 1; fi
source "$CANN_ENV"
PYTHON=${MLP_PYTHON:-python3}
VENV=${MLP_VENV:-$ROOT/.venv-mlp-npu}
"$PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$ROOT"
# Keep the generated CLI entry point, but prepare CANN for every qrun invocation.
cp "$VENV/bin/qrun" "$VENV/bin/qrun.python"
"$VENV/bin/python" - "$VENV" "$CANN_ENV" "$ROOT" <<'PY'
import pathlib,shlex,sys
v=pathlib.Path(sys.argv[1]); env=sys.argv[2]; root=sys.argv[3]
(v/'bin/qrun').write_text('#!/usr/bin/env bash\nset -eo pipefail\nsource '+shlex.quote(env)+'\nexport PYTHONPATH='+shlex.quote(root)+'${PYTHONPATH:+:"$PYTHONPATH"}\nexec '+shlex.quote(str(v/'bin/python'))+' '+shlex.quote(str(v/'bin/qrun.python'))+' "$@"\n')
PY
chmod +x "$VENV/bin/qrun"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" - "$ROOT" <<'PY'
import pathlib,sys,torch,torch_npu
import qlib
from qlib.contrib.model.pytorch_nn import DNNModelPytorch
from qlib.data._libs import rolling,expanding
assert pathlib.Path(qlib.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()), qlib.__file__
assert torch.npu.is_available(), 'NPU unavailable'
print('Qlib:',qlib.__file__)
print('NPU:',torch.npu.get_device_name(0))
PY
printf '\nActivate before running: source %q/bin/activate\n' "$VENV"
