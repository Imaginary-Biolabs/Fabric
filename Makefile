.PHONY: build publish-test publish clean

build:
	python -m pip install --upgrade build
	python -m build

publish-test: build
	python -m pip install --upgrade twine
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*

clean:
	rm -rf build dist *.egg-info


