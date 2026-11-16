set -x
cd <BENCHMARK_V2>
export PYTHONIOENCODING=utf-8
R=results/run_test100_20261018_0fd678a840bc
IN=inputs_test100_20261018.json; TG=targets_test100_20261018.json
BK=results/legend_availability.before_test100_20261018_0fd678a840bc.json
date
python run_astra100.py --inputs $IN --results $R --snapshot src_snapshots/0fd678a840bc/src --workers 3 --gpu-slots 2 --threads 4 --order-from results/run_synth/v46 > $R/v46_run.log 2>&1
date
python ensure_legends.py --inputs $IN --results $R/v46 > $R/ensure_legends.log 2>&1
cp $BK legend_availability.json
date
# v46 only: AI-ChartParser / LineEX were not run on this set (their columns read 0 / not_run)
python evaluate_entry.py --inputs $IN --targets $TG --results $R --ground-truth results/run_astra100_42377bd64ce7/empty_ground_truth.json > $R/evaluate_entry.log 2>&1
cp $BK legend_availability.json
date
echo ALL_DONE
