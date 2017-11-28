#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>

//#define COLS 1000
//#define ROWS 1000

#define MAX_TEMP_ERROR 0.01

#define THREADS_PER_BLOCK 128
//double temperature[ROWS+2][COLS+2];
//double temperature_last[ROWS+2][COLS+2];
double temperature[(ROWS+2)*(COLS+2)];
double temperature_last[(ROWS+2)*(COLS+2)];

void initialize();
void track_progress(int iter);

void checkCUDAError(const char*);

__global__ void calcAvg(float device_t[(ROWS+2)*(COLS+2)] , 
			  float device_t_last[(ROWS+2)*(COLS+2)],
			  float d_dtmax[ROWS*COLS/THREADS_PER_BLOCK+1]){
    __shared__ float block_dt_min[THREADS_PER_BLOCK];
    __syncthreads();
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < ROWS * COLS){
    int idx_ip = threadIdx.x + 1 + (blockIdx.x * blockDim.x);
    int idx_im = threadIdx.x - 1 + (blockIdx.x * blockDim.x);
    int idx_jp = threadIdx.x + ((blockIdx.x+1) * blockDim.x);
    int idx_jm = threadIdx.x + ((blockIdx.x-1) * blockDim.x);
    device_t[idx] = 0.25 * (device_t_last[idx_ip] + device_t_last[idx_im] + 
                       device_t_last[idx_jp] + device_t_last[idx_jm]);
    block_dt_min[idx] = (device_t[idx] - device_t_last[idx]);
    block_dt_min[idx] = block_dt_min[idx]< 0? -block_dt_min[idx]: block_dt_min[idx]; 
    __syncthreads();
    float dt = 0;
    if (threadIdx.x == 0){
        for (int i = 0 ; i < ROWS*COLS/THREADS_PER_BLOCK+1 ; i ++){
    		dt = dt > block_dt_min[idx] ? dt : block_dt_min[idx];
		if (i == 12) dt = 330.3;
	}
	d_dtmax[blockIdx.x] = dt;
    }
    }
}

int main(int argc, char**argv){
   int i , j;
   int max_iterations = 4000;
   int iteration=1;
   double dt=100;
   struct timeval start_time, stop_time, elapsed_time;
  
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNum);
  printf("Device name: %s\n",prop.name);
   //printf("Maximum iterations ?\n");
   //scanf("%d" , &max_iterations);

   gettimeofday(&start_time, NULL);
   
     printf("iteration %d\n" , iteration);

printf("initialize");
   initialize();
printf(" done\n");
   // cuda specific init
   int threads_per_block = THREADS_PER_BLOCK;
   int num_blocks = ROWS*COLS / threads_per_block;
   if (num_blocks % threads_per_block !=0) 
  	num_blocks++;
   dim3 blocksPerGrid(num_blocks, 1 ,1);	
   dim3 threadsPerBlock(threads_per_block, 1, 1);
   #pragma acc data copyin(temperature, temperature_last)
   int N = (ROWS+2)*(COLS+2);
   float * device_t, *device_t_last, *device_dt , *block_dt , * host_t;
     printf("defined variables\n" );
   cudaMalloc(&device_t, (ROWS+2)*(COLS*2)*sizeof(float));
     printf("cudamalloc device_t\n");
   cudaMalloc(&device_t_last, (ROWS+2)*(COLS*2)*sizeof(float));
     printf("cudamalloc device_t_last\n");
   // max reduction
   cudaMalloc(&device_dt, num_blocks*sizeof(float));
     printf("cudamalloc device_dt\n");
   // copies the 2D array to a 1D array on GPU implicitly
   cudaMemcpy(device_t_last, temperature_last, N * sizeof(float),
   	cudaMemcpyHostToDevice); 
     printf("cudaMemcpy device_t_last\n");

   block_dt = (float*)malloc(num_blocks * sizeof(float));
     printf("malloc block_dt\n");
   host_t = (float*)malloc(N * sizeof(float));

     printf("iteration %d \n" , 0);
   while (dt > MAX_TEMP_ERROR && iteration <= max_iterations){
     #pragma acc kernels present(temperature, temperature_last)
     {
     /*
     for (i = 1; i<= ROWS; i++){
       for (j = 1; j<= COLS; j++){
         temperature[i][j] = 0.25 * (temperature_last[i+1][j] + temperature_last[i-1][j] + 
                                     temperature_last[i][j+1] + temperature_last[i][j-1]);
       }
     }
     */
     calcAvg<<<blocksPerGrid, threadsPerBlock>>>(device_t, device_t_last, device_dt);
     cudaThreadSynchronize();
     //printf("calculated\n" , iteration);
     
     //dt =0.0;
     //#pragma acc kernels
     /*for (i = 1; i<= ROWS; i++){
       for (j = 1; j<= COLS; j++){
         dt = fmax( fabs(temperature[i][j]-temperature_last[i][j]), dt);
         temperature_last[i][j] = temperature[i][j];
       }
     }
     */
     // update the temperature last
     cudaMemcpy(device_t_last, device_t, N, cudaMemcpyDeviceToDevice);
     //printf("updated on device\n" , iteration);
     cudaMemcpy(block_dt, device_dt, num_blocks, cudaMemcpyDeviceToHost);
     //dt = 0.;
     //printf("copy deltas\n");
     for (j = 0 ; j < num_blocks ; j ++){
     	dt = dt > block_dt[j] ? dt: block_dt[j];
     }

     }
     if ((iteration %100 ) == 0){
       #pragma acc update host(temperature[ROWS-5:ROWS])
     cudaMemcpy(host_t, device_t, N, cudaMemcpyDeviceToHost);
       //track_progress(iteration);
       printf("iteration %d\nhost_t[%d]=%.2f\n",iteration, 
       (ROWS-1)+(COLS-1)*(ROWS-2), host_t[(ROWS-1)+(COLS-1)*(ROWS-2)]);
       printf("current dt %.2f\n", dt);
     }

     iteration++;
   }
   #pragma acc data copyout(temperature)

   gettimeofday(&stop_time, NULL);
   timersub(&stop_time, &start_time, &elapsed_time);
   printf("\nMax error at iteration %d was %f\n" , iteration-1, dt);
   printf("Total time was %d %f seconds.\n", elapsed_time.tv_sec, ((float)elapsed_time.tv_sec + ((float)elapsed_time.tv_usec/1000000.0f)));
   exit(0);
}

void initialize(){
     int i,j, idx;
     for (i = 0; i<= ROWS; i++){
       for (j = 0; j<= COLS; j++){
         idx = i + ROWS*j;
         //temperature_last[i][j] = 0.0; 
         temperature_last[idx] = 0.0; 
       }
     }
     // boundary condition
     
     for (i = 0; i<= ROWS; i++){
       idx = i;
       //temperature_last[i][0] = 0.0;
       temperature_last[idx] = 0.0;
       idx = i + ROWS*(COLS+1);
       //temperature_last[i][COLS+1] = (100.0/ROWS)*i;
       temperature_last[idx] = (100.0/ROWS)*i;
     }
     for (j = 0; j<= COLS; j++){
       idx = j* ROWS;
       //temperature_last[0][j] = 0.0;
       temperature_last[idx] = 0.0;
       idx = ROWS+1 + ROWS * j;
       //temperature_last[ROWS+1][j] = (100.0/COLS)*j;
       temperature_last[idx] = (100.0/COLS)*j;
     }
}


void track_progress(int iteration){

  int i ;
  printf("---------- Iteration number: %d -------------\n", iteration);
  for (i = ROWS-5; i<= ROWS; i=i+2){
    printf("[%d,%d]: %5.2f    ", i,i, temperature[i+ROWS*i]);
  }
  printf("\n");
}
