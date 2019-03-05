
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define CONST_B 10000
#define CONST_M 50
#define CONST_C 1
#define CONST_H 28
#define CONST_W 28
#define CONST_K 5
#define K_SIZE 25
#define CONST_H_OUT 24

namespace mxnet
{
namespace op
{




__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */

    int b = blockIdx.x;
    //int m = blockIdx.y;
    int h = threadIdx.x % 28;
    int w = threadIdx.x / 28;

    __shared__ float local_x [28][28];
    __shared__ float local_k [CONST_M][CONST_K][CONST_K];
    float local_y = 0;
    //float local_y_arr [CONST_M][CONST_H_OUT][CONST_H_OUT];
    //load in local_X
    if (w*28 + h < 784)
        local_x[h][w] = x4d(b,0,h,w);

    //Load in k to local memory
    if (threadIdx.x < 1024) //compare to just checking threadIdx.x
        local_k [threadIdx.x/K_SIZE][(threadIdx.x%K_SIZE)/CONST_K][((threadIdx.x%K_SIZE)%CONST_K)] = k[threadIdx.x]; //k4d(threadIdx.x/K_SIZE,0,(threadIdx.x%K_SIZE)/CONST_K,((threadIdx.x%K_SIZE)%CONST_K));
        //k[threadIdx.x];
    if (threadIdx.x < 226){ //remaining parts
        local_k [(threadIdx.x+1024)/K_SIZE][((threadIdx.x+1024)%K_SIZE)/CONST_K][(((threadIdx.x+1024)%K_SIZE)%CONST_K)] = k[1024+threadIdx.x];//k4d((threadIdx.x+1024)/K_SIZE,0,((threadIdx.x+1024)%K_SIZE)/CONST_K,(((threadIdx.x+1024)%K_SIZE)%CONST_K));
    }   //k[1024 + threadIdx.x]; 
    
    __syncthreads();

    //int p, q;
    if (h < CONST_H_OUT && w < CONST_H_OUT) {
        for(int m = 0;  m < CONST_M;  ++m) {             // for each output feature map
            local_y = 0; //_arr[m][h][w] = 0;
            
            //for(int c = 0;  c < C; c++){         // sum over all input feature maps
                /*
                for(p = 0; p < CONST_K; ++p){         // KxK  filter
                    for(q = 0; q < CONST_K; ++q){
                        local_y += local_x[h+p][w+q] * local_k[m][p][q]; //k4d(m,0,p,q);
                    }
                }
                */
                ///*
                local_y += local_x[h+0][w+0] * local_k[m][0][0] + local_x[h+0][w+1] * local_k[m][0][1] + local_x[h+0][w+2] * local_k[m][0][2] + local_x[h+0][w+3] * local_k[m][0][3] +  local_x[h+0][w+4] * local_k[m][0][4];

                local_y += local_x[h+1][w+0] * local_k[m][1][0] + local_x[h+1][w+1] * local_k[m][1][1] + local_x[h+1][w+2] * local_k[m][1][2] + local_x[h+1][w+3] * local_k[m][1][3] + local_x[h+1][w+4] * local_k[m][1][4];

                local_y += local_x[h+2][w+0] * local_k[m][2][0] + local_x[h+2][w+1] * local_k[m][2][1] + local_x[h+2][w+2] * local_k[m][2][2] + local_x[h+2][w+3] * local_k[m][2][3] + local_x[h+2][w+4] * local_k[m][2][4];

                local_y += local_x[h+3][w+0] * local_k[m][3][0] + local_x[h+3][w+1] * local_k[m][3][1] + local_x[h+3][w+2] * local_k[m][3][2] + local_x[h+3][w+3] * local_k[m][3][3] + local_x[h+3][w+4] * local_k[m][3][4];

                local_y += local_x[h+4][w+0] * local_k[m][4][0] + local_x[h+4][w+1] * local_k[m][4][1] + local_x[h+4][w+2] * local_k[m][4][2] + local_x[h+4][w+3] * local_k[m][4][3] + local_x[h+4][w+4] * local_k[m][4][4];
                //*/
            //}
            y4d(b,m,h,w) = local_y; 
        }
    }
    /*
    if(h < H_out && w < W_out){
        for (int m = 0  m < M;  m++){
            y4d(b,m,h,w) = local_y_arr[m][h][w];
        }
    }
    */

    #undef y4d
    #undef x4d
    #undef k4d
}




/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {


    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
     cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    /*This was taken from cpu implimentation*/
    const int B = x.shape_[0];  //number of images in batch
    const int M = y.shape_[1];  //each output feature
    const int C = x.shape_[1];  //output feature map
    const int H = x.shape_[2];  //each output element
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    dim3 gridDim(B,1,1); //maximum number of blocks allowed for GPU  --ECB
    dim3 blockDim(1024,1,1); //should be dimentions of the pictures(ceiling to power of 2) W,H is the dimentions of 1 photo I think -ECB

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
