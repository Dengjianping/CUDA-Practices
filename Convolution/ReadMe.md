# Convolution

## Running Project
```
Compiled with Visual Studio 2015 community version
```

# Tips
## 1. About convlution explaination, this link is the best one I've ever seen.

[Convolution Explaination](https://www.zhihu.com/question/22298352)


## 2. Use share memory to boost the performance of convolution in a single block.
```cpp
extern __shared__ input[];
extern __shared__ kernel[];
```