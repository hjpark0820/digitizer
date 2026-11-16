set -x
cd <BENCHMARK_V2>
export PYTHONIOENCODING=utf-8
R=results/run_test100v2_gtswatch_external_20261116
IN=inputs_test100v2_20261025.json; TG=targets_test100v2_20261025.json
date
# legend crops = annotated (ground-truth) key boxes for every chart; v46 is not part of this condition
python make_gt_swatch_legends.py --inputs $IN --out $R/v46 > $R/make_legends.log 2>&1
# published methods on all 100 charts, same code, weights and device (cuda:0) as the 2026-10-25 run
python run_external_v2.py --manifest $IN --output $R/external --shared-legends-root $R/v46 > $R/external.log 2>&1
python run_lineex_paper.py --manifest $IN --output $R/external --shared-legends-root $R/v46 > $R/lineex_paper.log 2>&1
date
# same scorer; v46 rows read 'not_run' here (no v46 predictions in this folder)
python evaluate_entry.py --inputs $IN --targets $TG --results $R --ground-truth results/run_astra100_42377bd64ce7/empty_ground_truth.json > $R/evaluate_entry.log 2>&1
date
echo ALL_DONE
