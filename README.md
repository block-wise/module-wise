Experiments in Table 3

```
python3 resnet110-16modules-stl10.py -trt par -bas 64 -lbs 0.1 -lrt 0.002 -ing 0.1 -tra 1 -tau 1 -vta 1 -ne0 300 -see 0 1 2 3 4 -avg
```
```
python3 resnet110-8modules-stl10.py -trt par -bas 64 -lbs 0.1 -lrt 0.002 -ing 0.1 -tra 1 -tau 1 -vta 1 -ne0 300 -see 0 1 2 3 4 -avg
```
```
python3 resnet110-4modules-stl10.py -trt par -bas 64 -lbs 0.1 -lrt 0.002 -ing 0.1 -tra 1 -tau 1 -vta 1 -ne0 300 -see 0 1 2 3 4 5 6 -avg
```
```
python3 resnet110-2modules-stl10.py -trt par -bas 64 -lbs 0.1 -lrt 0.002 -ing 0.1 -tra 1 -tau 100 -vta 1 -ne0 300 -see 0 1 2 3 4 -avg
```

Experiments in Table 2 

```
python3 resnet101-2modules-cifar100.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -tra 0 -see 1 2 3 -avg
```
```
python3 resnet101-2modules-cifar100.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -uzt 1 -uzs 50 -see 1 2 3 -avg
```


Experiments in Table 1

```
python3 resnet152-4modules-tinyimagenet.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -tra 0 -see 6
```
```
python3 resnet152-4modules-tinyimagenet.py -cln 1CNN -trt par -ne0 300 -opt sgd -lrt 0.003 -bas 256 -lbs 0.1 -inn orthogonal -ing 0.1 -tra 1 -tau 1000000 -vta 1 -see 6
```



