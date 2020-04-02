#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
//#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;
const int MPPBIN=7;
typedef struct bin_t {
    particle_t* particles[MPPBIN];
//    int parid[MPPBIN];
    int nparts=0;
    int nearbins[9];
    int nnbins=0;
} bin_t;
typedef struct track_t{
    particle_t* particles[MPPBIN];
//    int parids[MPPBIN];
    int add=0;
}track_t;
//int *nbin;
track_t* track;
int num_bins_per_side;
double bin_size;
bin_t* bins;
int blks;
int binblks;
const int NUM_THREADS=160;

//using namespace thrust;
__global__ void rebin_gpu(particle_t* particles,int num_parts,int num_bins_per_side, bin_t* bins,double bin_size,track_t* track){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
 //   if (tid>=num_bins_per_side*num_bins_per_side){
 //       return;
 //   }
   // printf("blockIdx.x=%d\n",blockIdx.x);
   // printf("blockDim.x=%d\n",blockDim.x);
    if (tid>=num_bins_per_side*num_bins_per_side){
        return;
    }
    for (int i=0;i<bins[tid].nparts;i++){
        particle_t* particle=bins[tid].particles[i];
        int bin_row=particle->y/bin_size;
        int bin_col=particle->x/bin_size;
        int id = bin_row+bin_col*num_bins_per_side;
        if (id != tid){
            //int thisparid = bins[tid].parid[i];
            //nbin[thisparid]=id;
            track[tid].particles[track[tid].add]=particle;
           // track[tid].parids[track[tid].add]=thisparid;
            track[tid].add+=1;
            for (int j=i;j<bins[tid].nparts-1;j++){
                bins[tid].particles[j]=bins[tid].particles[j+1];
             //   bins[tid].parid[j]=bins[tid].parid[j+1];
            }
            bins[tid].nparts-=1;
            i--;
        }
    }
}


   /* if (tid==0){
        for (int i=0;i<num_bins_per_side*num_bins_per_side;i++){
            for (int j=0;j<track[i].add;j++){
                particle_t* particle=track[i].particles[j];
                int bin_row=particle->y/bin_size;
                int bin_col=particle->x/bin_size;
                int id=bin_row+bin_col*num_bins_per_side;
                bins[id].particles[bins[id].nparts]=particle;
                bins[id].parid[bins[id].nparts]=track[i].parids[j];
                bins[id].nparts+=1;
            }

        }
    }
*/
__global__ void rebinaddgpu(track_t* track,bin_t* bins,int num_bins_per_side,double bin_size){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid>=num_bins_per_side*num_bins_per_side){
        return;
    }
    for (int j=0;j<track[tid].add;j++){
        particle_t* particle = track[tid].particles[j];
        int bin_row=particle->y/bin_size;
        int bin_col=particle->x/bin_size;
        int id=bin_row+bin_col*num_bins_per_side;
        int nparts = atomicAdd(&(bins[id].nparts),1);
        bins[id].particles[nparts]=particle;
        //bins[id].parid[nparts]=track[tid].parids[j];
    }
}
__global__ void clear_gpu(track_t* track,int num_bins_per_side){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid>=num_bins_per_side*num_bins_per_side){
        return;
    }
    track[tid].add=0;
}
// Apply the force from neighbor to particle
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* parts, int num_parts,bin_t* bins,int num_bins_per_side) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_bins_per_side*num_bins_per_side){
        return;
    }
    //printf("griddim=%d\n",gridDim.y);
    bin_t* thisbin = &bins[tid];
    if (threadIdx.y>=thisbin->nparts){
        return;
    }
//    for (int i=0;i<thisbin->nparts;i++){
    particle_t* particle=thisbin->particles[threadIdx.y];
    particle->ax = 0;
    particle->ay = 0;
    for (int j=0;j<thisbin->nnbins;j++){
        bin_t* nearbins=&bins[thisbin->nearbins[j]];
        for (int k=0;k<nearbins->nparts;k++){
            particle_t* neaparticle = nearbins->particles[k];
            apply_force_gpu(*particle,*neaparticle);
        }
    }
//}
}

// Integrate the ODE
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
//    for (int i=tid;i<num_parts;i+=)
    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

//device_vector<> *collision_bins;  // TODO: Rename to adjacent_bins.
//device_vector<device_vector<particle_t*>*> *track;
// Make bins .1% larger to account for rounding error in size computation.
// TODO: How does this affect scaling?
#define OVERFILL 1.001

double ceil(double x) {
    return (double)(x + 1);
}

double round(double x) {
    return (double)(x + 0.5);
}

__global__ void binparticles(particle_t* parts,int num_parts,bin_t* bins,int num_bins_per_side,double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }
        particle_t *p = parts + tid;

        int bin_row = p->y / bin_size; // Floor.
        int bin_col = p->x / bin_size; // Floor.

        int id = bin_row+bin_col*num_bins_per_side;

        int pnparts=atomicAdd(&(bins[id].nparts),1);
        //nbin[tid]=id;
        bins[id].particles[pnparts]=p;
        //bins[id].parid[pnparts]=tid;
}
__global__ void setupnearbins(bin_t* bins,int num_bins_per_side){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid>=num_bins_per_side*num_bins_per_side){
        return;
    }
    
    int i = tid%num_bins_per_side;
    int j = tid/num_bins_per_side;
    for (int k = i - 1; k < i + 2; k++) {
        for (int l = j - 1; l < j + 2; l++) {
            if (k >= 0 && l >= 0 && k < num_bins_per_side && l < num_bins_per_side) {
                int nid = k+l*num_bins_per_side;  
     	        bins[tid].nearbins[bins[tid].nnbins]=nid;
                bins[tid].nnbins+=1;
            }
        }
    }
}
void init_simulation(particle_t* parts, int num_parts, double size) {
    //cout << "Initializing simulation...\n";

        // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here.

    // We will divide the simulation environment into n 2D bins because we can assume
    // that the simulator is sparse and there will be at most n interactions at each step
    // of the simulation.

        cout<<"start ini\n";
        cout.flush();
    num_bins_per_side = ceil(sqrt(num_parts)); // Ceil.
    bin_size = (size / num_bins_per_side) * OVERFILL;

    // Create bins.
    //cout << "Creating bins...\n";
    blks=(num_parts+NUM_THREADS-1)/NUM_THREADS;
    binblks=(num_bins_per_side*num_bins_per_side+NUM_THREADS-1)/NUM_THREADS;
    cout<<"binblks="<<binblks<<endl;
    cout<<"blks="<<blks<<endl;
    cudaMalloc(&bins,num_bins_per_side*num_bins_per_side*sizeof(bin_t));
    //cudaMalloc(&nbin,num_parts*sizeof(int));
    cudaMalloc(&track,num_bins_per_side*num_bins_per_side*sizeof(track_t));
    cudaError_t cudaerr=cudaGetLastError();
    if (cudaerr != cudaSuccess)
      printf("malloc failed\"%s\".\n",
               cudaGetErrorString(cudaerr));
    // Bin particles.
    //cout << "Binning particles...\n";
    binparticles<<<blks,NUM_THREADS>>>(parts,num_parts,bins,num_bins_per_side,bin_size);
    cudaerr=cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("binpart\"%s\".\n",
               cudaGetErrorString(cudaerr));
    setupnearbins<<<binblks,NUM_THREADS>>>(bins,num_bins_per_side);
    cudaerr=cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("setupnearbins\"%s\".\n",
               cudaGetErrorString(cudaerr));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    static int loop = 0;

//    auto start_time = chrono::steady_clock::now();
   
    rebin_gpu<<<binblks,NUM_THREADS>>>(parts,num_parts,num_bins_per_side,bins,bin_size,track);    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        cout<<"rebingpu"<<loop<<":"<<cudaGetErrorString(cudaerr)<<endl;
//    auto end_time = chrono::steady_clock::now();
//    chrono::duration<double> diff = end_time - start_time;
//    double seconds = diff.count();
//    cout << "Rebin kernel took " << seconds << " seconds." << endl;

//    start_time = chrono::steady_clock::now();
    rebinaddgpu<<<binblks,NUM_THREADS>>>(track,bins,num_bins_per_side,bin_size);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        cout<<"rebinaddgpu"<<loop<<":"<<cudaGetErrorString(cudaerr)<<endl;
//    end_time = chrono::steady_clock::now();
//    diff = end_time - start_time;
//    seconds = diff.count();
//    cout << "Rebinadd kernel took " << seconds << " seconds." << endl;

//    start_time = chrono::steady_clock::now();
    clear_gpu<<<binblks,NUM_THREADS>>>(track,num_bins_per_side);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        cout<<"cleargpu"<<loop<<":"<<cudaGetErrorString(cudaerr)<<endl;
//    end_time = chrono::steady_clock::now();
//    diff = end_time - start_time;
//    seconds = diff.count();
//    cout << "Clear kernel took " << seconds << " seconds." << endl;

//    start_time = chrono::steady_clock::now();
    dim3 block_dim=dim3(NUM_THREADS,6,1);
    dim3 grid_dim=dim3(binblks,1,1);
    compute_forces_gpu<<<grid_dim,block_dim>>>(parts,num_parts,bins,num_bins_per_side);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        cout<<"forcegpu"<<loop<<":"<<cudaGetErrorString(cudaerr)<<endl;
//    end_time = chrono::steady_clock::now();
//    diff = end_time - start_time;
//    seconds = diff.count();
//    cout << "Compute_forces kernel took " << seconds << " seconds." << endl;

//    start_time = chrono::steady_clock::now();
    move_gpu<<<binblks,NUM_THREADS>>>(parts, num_parts,size);  // NOTE(vsatish): Changed this from blks.
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        cout<<"movegpu"<<loop<<":"<<cudaGetErrorString(cudaerr)<<endl;
//    end_time = chrono::steady_clock::now();
//    diff = end_time - start_time;
//    seconds = diff.count();
//    cout << "Move kernel took " << seconds << " seconds." << endl;

    loop++;
    // Re-bin particles.
  //  rebin<<<blks,NUM_THREAD>>>(bins,num_bins_per_side,collision_bins);
   /* for (int i = 0; i < num_bins_per_side; i++) {
        for (int j = 0; j < num_bins_per_side; j++) {
            device_vector<particle_t*> *particles = (*((*bins)[i]))[j]->particles;  // TODO: Does saving this cause iteration issues?
            for (int k = 0; k < particles->size(); k++) {
                particle_t *p = (*particles)[k];

                int bin_row = p->y / bin_size; // Floor.
                int bin_col = p->x / bin_size; // Floor.

                if (bin_row != i || bin_col != j) {
                    // Remove from current bin.
                    particles->erase(particles->begin()+k);
                    k--;
                    int ID=omp_get_thread_num();
                    (*track)[ID]->push_back(p);
                    // Move to correct bin.
                   // (*((*bins)[bin_row]))[bin_col]->particles->push_back(p);
                }
            }
        }
    }
      for (int i=0;i<Nthreads;i++){
        device_vector<particle_t*> *particles =(*track)[i];
        for (int m=0;m<particles->size();m++){
          particle_t *p =(*particles)[m];
          int bin_row=p->y/bin_size;
          int bin_col=p->x/bin_size;
          (*((*bins)[bin_row]))[bin_col]->particles->push_back(p);
        }
        (*track)[i]->clear();
      }

}*/
}
// Free memory.

