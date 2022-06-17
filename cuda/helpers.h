#ifndef HELPERS_H
#define HELPERS_H

    #define MAX(a,b) (a < b)?b:a

    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);

    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \

    #define TIMEELAPSED(label) time##label;

    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            printf("Cuda error: %s: %s, line %d\n", cudaGetErrorString(err),   \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        } else {                                                               \
            printf("Cuda kernel launched successfully\n");                     \
        }                                                                      \
    }

    // transfer constants
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
    #define H2H (cudaMemcpyHostToHost)
    #define D2D (cudaMemcpyDeviceToDevice)

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

#endif
