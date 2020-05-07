MODULENAME = Hyperparameter_Optimization 

help:
	@echo ""
	@echo "Welcome to my project!!!"
	@echo "To get started create an environment using:"
	@echo "	make init"
	@echo "	conda activate masterthesis"
	@echo ""
	@echo "To deactivate the environment:"
	@echo " conda deactivate"
	@echo ""
	@echo "Update the environment:"
	@echo "	conda env export > ./Software/environment.yml "
	@echo ""


init:
	conda env create --name masterthesis --file ./Software/environment.yml


env:
	conda env export > ./Software/environment.yml

clean:
	rm .gitkeep -f
	rm .DS_Store -f

.PHONY: init doc lint test 

