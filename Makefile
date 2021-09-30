clean:
	rm -rf dist build stamp_processing.egg-info

build:
	pip install --upgrade build
	python -m build

pypi: 
	pip install --upgrade twine
	twine upload --repository testpypi dist/*

