train-fashion:
	python3 Train.py -M faC_c2f2

train-sop:
	python3 Train.py -M sopE_res18

attack-fashion:
	bash bin/faspsa 5 50 4

attack-sop:
	bash bin/spsa 5 50 4

perf-farand:
	python3 -m cProfile -s cumulative BlackOA.py -M Fashion -D cuda -A RandSearch -e .0156862 -N 99 -k 5 -c 50 -P 50
perf-fabatk:
	python3 -m cProfile -s cumulative BlackOA.py -M Fashion -D cuda -A Batk -e .0156862 -N 99 -k 5 -c 50 -P 1
perf-fapso:
	python3 -m cProfile -s cumulative BlackOA.py -M Fashion -D cuda -A PSO -e .0156862 -N 99 -k 5 -c 50 -P 40
perf-fanes:
	python3 -m cProfile -s cumulative BlackOA.py -M Fashion -D cuda -A NES -e .0156862 -N 99 -k 5 -c 50 -P 1
perf-faspsa k="5":
	python3 -m cProfile -s cumulative BlackOA.py -M Fashion -D cuda -A SPSA -e .0156862 -N 99 -k "{{k}}" -c 50 -P 1
perf-rand:
	python3 -m cProfile -s cumulative BlackOA.py -M Sop -D cuda -A RandSearch -e .0156862 -N 99 -k 5 -c 50 -P 50
perf-batk:
	python3 -m cProfile -s cumulative BlackOA.py -M Sop -D cuda -A Batk -e .0156862 -N 99 -k 5 -c 50 -P 1
perf-pso:
	python3 -m cProfile -s cumulative BlackOA.py -M Sop -D cuda -A PSO -e .0156862 -N 99 -k 5 -c 50 -P 40
perf-nes:
	python3 -m cProfile -s cumulative BlackOA.py -M Sop -D cuda -A NES -e .0156862 -N 99 -k 5 -c 50 -P 1
perf-spsa k="5":
	python3 -m cProfile -s cumulative BlackOA.py -M Sop -D cuda -A SPSA -e .0156862 -N 99 -k "{{k}}" -c 50 -P 1
transfer:
	python3 Attack.py -M faC_c2f2 -A SPO:PGD-M5 -v -T faC_lenet:trained/faC_lenet.sdth
practical:
	python3 PracticalOA.py -l lib/t8.png -Q 100
jd:
	python3 PracticalOA.py -Q 100 --randperm -k 10 -l $(python3 bin/randsop.py)
