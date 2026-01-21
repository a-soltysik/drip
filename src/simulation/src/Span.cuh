#pragma once

#include <thrust/device_vector.h>

namespace drip::sim
{

template <typename T>
struct Span
{
    T* data;
    size_t size;

    __host__ __device__ auto operator[](size_t idx) -> T&
    {
        return data[idx];
    }

    __host__ __device__ auto operator[](size_t idx) const -> const T&
    {
        return data[idx];
    }

    static auto fromDeviceVector(thrust::device_vector<T>& vec) -> Span
    {
        return Span {.data = thrust::raw_pointer_cast(vec.data()), .size = vec.size()};
    }
};

}