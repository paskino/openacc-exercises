#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

//#define COLS 1000
//#define ROWS 1000

#define MAX_TEMP_ERROR 0.01

double temperature[ROWS+2][COLS+2];
double temperature_last[ROWS+2][COLS+2];

void initialize();
void track_progress(int iter);

void main(int argc, char**argv){
   int i , j;
   int max_iterations = 4000;
   int iteration=1;
   double dt=100;
   struct timeval start_time, stop_time, elapsed_time;
   
   //printf("Maximum iterations ?\n");
   //scanf("%d" , &max_iterations);

   gettimeofday(&start_time, NULL);

   initialize();
   #pragma acc data copyin(temperature, temperature_last)
   while (dt > MAX_TEMP_ERROR && iteration <= max_iterations){
     #pragma acc kernels present(temperature, temperature_last)
     {
     for (i = 1; i<= ROWS; i++){
       for (j = 1; j<= COLS; j++){
         temperature[i][j] = 0.25 * (temperature_last[i+1][j] + temperature_last[i-1][j] + 
                                     temperature_last[i][j+1] + temperature_last[i][j-1]);
       }
     }
     dt =0.0;
     //#pragma acc kernels
     for (i = 1; i<= ROWS; i++){
       for (j = 1; j<= COLS; j++){
         dt = fmax( fabs(temperature[i][j]-temperature_last[i][j]), dt);
         temperature_last[i][j] = temperature[i][j];
       }
     }
     }
     if ((iteration %100 ) == 0){
       #pragma acc update host(temperature[ROWS-5:ROWS])
       track_progress(iteration);
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
     int i,j;
     for (i = 0; i<= ROWS; i++){
       for (j = 0; j<= COLS; j++){
         temperature_last[i][j] = 0.0; 
       }
     }
     // boundary condition
     
     for (i = 0; i<= ROWS; i++){
       temperature_last[i][0] = 0.0;
       temperature_last[i][COLS+1] = (100.0/ROWS)*i;
     }
     for (j = 0; j<= COLS; j++){
       temperature_last[0][j] = 0.0;
       temperature_last[ROWS+1][j] = (100.0/COLS)*j;
     }
}


void track_progress(int iteration){

  int i ;
  printf("---------- Iteration number: %d -------------\n", iteration);
  for (i = ROWS-5; i<= ROWS; i=i+2){
    printf("[%d,%d]: %5.2f    ", i,i, temperature[i][i]);
  }
  printf("\n");
}
