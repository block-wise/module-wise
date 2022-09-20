To reproduce results from Table 1 run
python3 resnet152-4modules-tinyimagenet.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -tra 0 -see 6
python3 resnet152-4modules-tinyimagenet.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -tra 1 -tau 1000000 -vta 1 -see 6
python3 resnet152-e2e-tinyimagenet.py 

To reproduce results from Table 2 run
python3 resnet101-2modules-cifar100.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -tra 0 -see 1 2 3 -avg
python3 resnet101-2modules-cifar100.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -uzt 1 -uzs 50 -see 1 2 3 -avg
python3 resnet101-e2e-cifar100.py

