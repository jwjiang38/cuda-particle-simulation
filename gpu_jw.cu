#include "common.h"
#include <cmath>
#include <iostream>
//#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//TODO:Use vector
using namespace std;
const int MPPBIN=5;
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
    if (tid==0){
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

__global__ void binparticles(particle_t* parts,int num_parts,bin_t* bins,int num_bins_per_side,int* nbin,double bin_size){
    for (int i = 0; i < num_parts; i++) {
        particle_t *p = parts + i;

        int bin_row = p->y / bin_size; // Floor.
        int bin_col = p->x / bin_size; // Floor.

        int id = bin_row+bin_col*num_bins_per_side;
        bins[id].particles[bins[id].nparts]=p;
        bins[id].parid[bins[id].nparts]=i;
        bins[id].nparts+=1;
        nbin[i]=id;
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
    binparticles<<<1,1>>>(parts,num_parts,bins,num_bins_per_side,nbin,bin_size);

    // Next we will group adjacent bins together that have possibilities of collisions.
    // There will be 9 bins in each group. We will expect there to be on average one
    // collision in each group. This is how we will limit the complexity of the problem.
    //cout << "creating track variable\n";
/*    track=new device_vector<device_vector<particle_t*>*>;
    for (int i =0 ; i< omp_get_num_procs();i++){
        track->push_back(new device_vector<particle_t*>());
    }*/
    //cout << "Creating collision bins...\n";
   // collision_bins = new device_vector<device_vector<device_vector<bin_t*>*>*>();  // TODO: Having nested pointers is stupid in this case (even the top-most pointer isn't needed).
    for (int i = 0; i < num_bins_per_side; i++) {

        for (int j = 0; j < num_bins_per_side; j++) {
            int id = i+j*num_bins_per_side;
            for (int k = i - 1; k < i + 2; k++) {
                for (int l = j - 1; l < j + 2; l++) {
                    if (k >= 0 && l >= 0 && k < num_bins_per_side && l < num_bins_per_side) {
                            int nid = k+l*num_bins_per_side;  
			    bins[id].nearbins[bins[id].nnbins]=nid;
                            bins[id].nnbins+=1;
  // NOTE: Indexing must be UD/LR otherwise correctness against ref implementation will fail.
                    }
                }
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

/*    for (int i = 0; i < num_bins_per_side; i++) {
        for (int j = 0; j < num_bins_per_side; j++) {
            bin_t *bin = (*((*bins)[i]))[j];
            device_vector<particle_t*> *particles = bin->particles;
            device_vector<bin_t*> *adjacent_bins = (*((*collision_bins)[i]))[j];

            for (int k = 0; k < particles->size(); k++) {
                particle_t *particle = (*particles)[k];
                particle->ax = particle->ay = 0;  // TODO: Handle case where there are no neighbors and setting this would be wrong.
                                                  // NOTE: Reference implementation also makes this error so...

                for (int l = 0; l < adjacent_bins->size(); l++) {
                    device_vector<particle_t*> *colliding_particles = (*adjacent_bins)[l]->particles;

                    for (int m = 0; m < colliding_particles->size(); m++) {
                        particle_t *colliding_particle = (*colliding_particles)[m];

                        apply_force(*particle, *colliding_particle);
                    }
                }
            }
        }
    }
*/
    // Move particles.
        move_gpu<<<blks,NUM_THREADS>>>(parts, num_parts,size);
        rebin_gpu<<<binblks,NUM_THREADS>>>(parts,num_parts,num_bins_per_side,nbin,bins,bin_size,track);
        clear_gpu<<<binblks,NUM_THREADS>>>(track,num_bins_per_side);
        compute_forces_gpu<<<blks,NUM_THREADS>>>(parts,num_parts,nbin,bins);
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

