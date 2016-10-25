# Bank Confilct
1. Ensure your device warp size
```cpp
cudaDeviceProp *prop;
cudaGetDeviceProperties(prop, 0); // suppose your computer just has only one device
int warp = prop->warpSize; // generally, it should be 32
```