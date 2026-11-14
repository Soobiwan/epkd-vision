#!/usr/bin/env bash
# Run on the TPU VM from imagenet_colab:  bash bench/tpu_check_fixes.sh
# Checks (1) flat-H2D at crops %8 != 0 and (2) UPDATES_OK/LOST per crop and compile scope; (3) step equivalence at crop 140.
cd "$(dirname "$0")/.." || exit 1
L=/content/work/logs; mkdir -p "$L"
for scope in step fwdbwd; do
  python3 bench/tpu_probe3.py --device tpu --steps 20 --crops 112,140,168,196,224 --compile-scope $scope --h2d 0 --precompute 0 \
    > "$L/check_probe_$scope.log" 2>&1
  echo "== probe $scope"; grep -E "UPDATES_|ERROR|batch_transfer failed" "$L/check_probe_$scope.log"
done
for dt in bf16 fp32; do
  python3 tests/test_step_equivalence.py --device tpu --dtype $dt --crop 140 --anchor 96 --compile-scope none \
    > "$L/check_equiv_$dt.log" 2>&1
  echo "== equivalence $dt"; tail -n 3 "$L/check_equiv_$dt.log"
done
