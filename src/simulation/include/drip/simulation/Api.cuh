#pragma once

#ifdef _WIN32
#    ifdef DRIP_CUDA_EXPORTS
#        define DRIP_CUDA_API __declspec(dllexport)
#    else
#        define DRIP_CUDA_API __declspec(dllimport)
#    endif
#else
#    define DRIP_CUDA_API
#endif
