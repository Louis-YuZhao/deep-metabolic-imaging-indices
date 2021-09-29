# deep-metabolic-imaging-biomarker
Code for paper titled "Differential diagnosis of parkinsonism with deep metabolic imaging biomarker" 

by Yu Zhao, University of Bern and Technical University of Munich 

last modified 07.21.2020

Requirement:
  Python 2.7.3, 
  tensorflow 1.9.0, 
  Keras 2.2.2


Guideline for utilizing (Pegasus):

(1) pull docker image from http://mirrors.tencent.com/

    docker pull mirrors.tencent.com/single_cell_analysis/pegasus_yaml:v1

(2) put the dataset into a defined folder
    
    for example: "/lvm_data/dataset/test_data_first_batch"
    the MantonBM_nonmix_subset.zarr.zip from https://storage.googleapis.com/terra-featured-workspaces/Cumulus/MantonBM_nonmix_subset.zarr.zip. was utilized for testing the framework.

(3) specify the yaml config documents of each step:

    sample yaml files are provided in the ./single_cell_analysis_framework/RawData:
    for preprocessing: pegasus_preprocessing.yaml
    for clusting: pegasus_clusting.yaml
    for differential expression: pegasus_differential_expression.yaml
    etc.

(4) indicate input path, output path, and the specified yaml-config file of each step in an .sh document:

    Take clustering-step as an example: The detailed information are defined in run_clusting.sh

    python ./SingleCell/app/TencentSCFApp.py \ # the main founction
        --specification_path='./RawData/pegasus_clusting.yaml' \ #yaml config file 
        --result_path='./Results/Test_wdl_multi_docker' \ # result path
        --input_path='./Results/Test_wdl_multi_docker/Preprocessing_result.h5ad' # input path


(5) indicate the docker image and the processing order of each step in run_wdl_multi_docker.sh file:

    for instance:
    
    # (1) step1 preprocessing
    
    docker run -it --rm \
    -v /lvm_data/dataset/test_data_first_batch:/Input:rw \ 
    -v /aaa/louisyuzhao/project2/singlecell/single_cell_analysis_framework:/algorithm:rw \
    mirrors.tencent.com/single_cell_analysis/pegasus_yaml:v1 \
    bash ./algorithm/run_preprocessing.sh

    # (2) step2 clusting
    
    docker run -it --rm \
    -v /lvm_data/dataset/test_data_first_batch:/Input:rw \
    -v /aaa/louisyuzhao/project2/singlecell/single_cell_analysis_framework:/algorithm:rw \
    mirrors.tencent.com/single_cell_analysis/pegasus_yaml:v1 \
    bash ./algorithm/run_clusting.sh

    # (3) step3 differential_expression
    
    docker run -it --rm \
    -v /lvm_data/dataset/test_data_first_batch:/Input:rw \
    -v /aaa/louisyuzhao/project2/singlecell/single_cell_analysis_framework:/algorithm:rw \
    mirrors.tencent.com/single_cell_analysis/pegasus_yaml:v1 \
    bash ./algorithm/run_differential_expression.sh

(6) run run_wdl_multi_docker.sh
    
    bash ./XXX/run_wdl_multi_docker.sh
