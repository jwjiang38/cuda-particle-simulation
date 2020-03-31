#include "common.h"
#include <cmath>
#include <iostream>
//#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;
const int MPPBIN=7;
typedef struct bin_t {
    particle_t* particles[MPPBIN];
    int parid[MPPBIN];
    int nparts=0;
    int nearbins[9];
    int nnbins=0;
} bin_t;
typedef struct track_t{
    particle_t* particles[MPPBIN];
    int parids[MPPBIN];
    int add=0;
}track_t;
int *nbin;
track_t* track;
int num_bins_per_side;
double bin_size;
bin_t* bins;
int blks;
int binblks;
const int NUM_THREADS=256;

using namespace thrust;
__global__ void rebin_gpu(particle_t* particles,int num_parts,int num_bins_per_side,int* nbin, bin_t* bins,double bin_size,track_t* track){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid>=num_bins_per_side*num_bins_per_side){
        return;
    }
    for (int i=0;i<bins[tid].nparts;i++){
        particle_t* particle=bins[tid].particles[i];
        int bin_row=particle->y/bin_size;
        int bin_col=particle->x/bin_size;
        int id = bin_row+bin_col*num_bins_per_side;
        if (id != tid){
            int thisparid = bins[tid].parid[i];
            nbin[thisparid]=id;
            track[tid].particles[track[tid].add]=particle;
            track[tid].parids[track[tid].add]=thisparid;
            track[tid].add+=1;
            for (int j=i;j<bins[tid].nparts-1;j++){
                bins[tid].particles[j]=bins[tid].particles[j+1];
                bins[tid].parid[j]=bins[tid].parid[j+1];
            }
            bins[tid].nparts-=1;
            i--;
        }
    }
    __syncthreads();

               
    for (int j=0;j<track[tid].add;j++){
        particle_t* particle = track[tid].particles[j];
        int bin_row=particle->y/bin_size;
        int bin_col=particle->x/bin_size;
        int id=bin_row+bin_col*num_bins_per_side;
        int nparts = atomicAdd(&(bins[id].nparts),1);
        bins[id].particles[nparts]=particle;
        bins[id].parid[nparts]=track[tid].parids[j];
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
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* parts, int num_parts,int* nbin,bin_t* bins) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }


    parts[tid].ax = parts[tid].ay = 0;
    
    for (int i=0;i<bins[nbin[tid]].nnbins;i++){
        bin_t nearbins=bins[bins[nbin[tid]].nearbins[i]];
        for (int j=0;j<nearbins.nparts;j++){
            particle_t* particle = nearbins.particles[j];
            apply_force_gpu(parts[tid],*particle);
        }
    }
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

#define OVERFILL 1.001

double ceil(double x) {
    return (double)(x + 1);
}

double round(double x) {
    return (double)(x + 0.5);
}

__global__ void binparticles(particle_t* parts,int num_parts,bin_t* bins,int num_bins_per_side,int* nbin,double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }
        particle_t *p = parts + tid;

        int bin_row = p->y / bin_size; // Floor.
        int bin_col = p->x / bin_size; // Floor.

        int id = bin_row+bin_col*num_bins_per_side;

        int pnparts=atomicAdd(&(bins[id].nparts),1);
        nbin[tid]=id;
        bins[id].particles[pnparts]=p;
        bins[id].parid[pnparts]=tid;
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
    cudaMallocManaged(&bins,num_bins_per_side*num_bins_per_side*sizeof(bin_t));
    cudaMallocManaged(&nbin,num_parts*sizeof(int));
    cudaMallocManaged(&track,num_bins_per_side*num_bins_per_side*sizeof(track_t));
    // Bin particles.
    //cout << "Binning particles...\n";
    binparticles<<<blks,NUM_THREADS>>>(parts,num_parts,bins,num_bins_per_side,nbin,bin_size);
    setupnearbins<<<binblks,NUM_THREADS>>>(bins,num_bins_per_side);

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Move particles.
        move_gpu<<<blks,NUM_THREADS>>>(parts, num_parts,size);
        rebin_gpu<<<binblks,NUM_THREADS>>>(parts,num_parts,num_bins_per_side,nbin,bins,bin_size,track);
        clear_gpu<<<binblks,NUM_THREADS>>>(track,num_bins_per_side);
        compute_forces_gpu<<<blks,NUM_THREADS>>>(parts,num_parts,nbin,bins);
