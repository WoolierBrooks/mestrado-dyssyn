# Create environment
setup:
	conda env create -f environment.yml

# Run experiment
run:
	python src/experiments/run_experiment.py --config=$(config)

# Clean temporary files
clean:
	find . -name "__pycache__" -type d -exec rm -r {} +
