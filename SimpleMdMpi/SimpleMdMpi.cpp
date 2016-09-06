#include "mpi/mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#pragma comment (lib, "msmpi.lib") 

using namespace std;
#define TIMES 1000
const int N = 6400;         // number of particles
double r[N][3];           // positions
double v[N][3];           // velocities
double a[N][3];           // accelerations
double tmp[N][3];
double L = 10;            // linear size of cubical volume
double vMax = 0.1;        // maximum initial velocity component

#define ABC

void initialize() {

    int n = int(ceil(pow(N, 1.0 / 3)));  // number of atoms in each direction
    double a = L / n;                  // lattice spacing
    int p = 0;                         // particles placed so far
    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            for (int z = 0; z < n; z++) {
                if (p < N) {
                    r[p][0] = (x + 0.5) * a;
                    r[p][1] = (y + 0.5) * a;
                    r[p][2] = (z + 0.5) * a;
                }
                ++p;
            }
    // initialize velocities
    for (int p = 0; p < N; p++){
        for (int i = 0; i < 3; i++){
            v[p][i] = vMax * (2 * rand() / double(RAND_MAX) - 1);
        }
    }
}
void SingleAcclerations(int i)
{
    for (int j = i + 1; j < N; j++) {
        double rij[3];               // position of i relative to j
        double rSqd = 0;
        for (int k = 0; k < 3; k++) {
            rij[k] = r[i][k] - r[j][k];
            rSqd += rij[k] * rij[k];
        }
        double f = 24 * (2 * pow(rSqd, -7) - pow(rSqd, -4));
        for (int k = 0; k < 3; k++) {
            a[i][k] += rij[k] * f;
            a[j][k] -= rij[k] * f;
        }
    }
}
//map
void computeAccelerations(int index, int from, int to) {
    memset(&a[0][0], 0, N * 3 * sizeof(double));
    //cout << index << ":(" << from << "," << to << ")" << endl;
//#pragma omp parallel for
    for (int i = from; i < to; i++){    // loop over all distinct pairs i,j
        SingleAcclerations(i);
    }
}

double instantaneousTemperature() {
    double sum = 0;
    for (int i = 0; i < N; i++){
        for (int k = 0; k < 3; k++){
            sum += v[i][k] * v[i][k];
        }
    }
    return sum / (3 * (N - 1));
}

void velocityVerlet(double dt) {
    computeAccelerations(0, 0, N - 1);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            r[i][k] += v[i][k] * dt + 0.5 * a[i][k] * dt * dt;
            v[i][k] += 0.5 * a[i][k] * dt;
        }
    }
    computeAccelerations(0, 0, N - 1);
    for (int i = 0; i < N; i++){
        for (int k = 0; k < 3; k++){
            v[i][k] += 0.5 * a[i][k] * dt;
        }
    }
}

void sendBlock(int numprocs)
{
    long area = N * (N - 1) / 2;
    long splice = area / (numprocs - 1);

    long from = N;
    long to = 0;
    long sum = 0;
    int index = 1;
    for (int j = N - 1; j > 0; j--){
        sum += j;
        if (sum >= splice || j == 1){
            to = j;
            long fromto[2] = { N - from, N - to };
            MPI_Send(&fromto[0], 2, MPI_LONG, index, 2, MPI_COMM_WORLD);
            from = j;
            sum = 0;
            index++;
        }
    }
}

void reduceBlock(int numprocs)
{
    memset(a, 0, N * 3 * sizeof(double));
    for (int i = 1; i < numprocs; i++){
        double tmp[N][3] = { 0 };
        MPI_Status status;
        MPI_Recv(&tmp, 3 * N, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, &status);
        //cout << i <<"a received"<< endl;
        for (int j = 0; j < N; j++){
            for (int k = 0; k < 3; k++){
                //printf("%d[%d]: %f\n", i, j, tmp[j][k]);
                a[j][k] += tmp[j][k];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    int myid, numprocs;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    double dt = 0.01;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Get_processor_name(processor_name, &namelen);
    initialize();
    if (numprocs == 1){
	cout << "N = " << N << endl;
	double t1 = MPI_Wtime();
	ofstream file("NoMPI.data");
        for (int times = 0; times < TIMES; times++){
            if(times == 0)
		computeAccelerations(0, 0, N - 1);
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < 3; k++) {
                    r[i][k] += v[i][k] * dt + 0.5 * a[i][k] * dt * dt;
                    v[i][k] += 0.5 * a[i][k] * dt;
                }
            }
            computeAccelerations(0, 0, N - 1);
            for (int i = 0; i < N; i++){
                for (int k = 0; k < 3; k++){
                    v[i][k] += 0.5 * a[i][k] * dt;
                }
            }
            file << instantaneousTemperature() << "\n";
        }
        file.close();
        double t2 = MPI_Wtime();
	ofstream file2("single.time");
	cout << "timespan:" << t2 - t1 << "s" << endl;
	file2 << t2 - t1 << endl;
	file2.close();
	return 0;
    }
    if (myid == 0){
        cout << "N = " << N << endl;
	cout << "numprocs = " << numprocs << endl;
	//int num = numprocs - 1;
        double t1 = MPI_Wtime();
	ofstream file("mpi.data");
        for (int i = 1; i < numprocs; i++){
            MPI_Send(&r[0][0], N * 3, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }
        //send each block
        sendBlock(numprocs);
        //REDUCE
        reduceBlock(numprocs);
        for (int times = 0; times < TIMES; times++){
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < 3; k++) {
                    r[i][k] += v[i][k] * dt + 0.5 * a[i][k] * dt * dt;
                    v[i][k] += 0.5 * a[i][k] * dt;
                }
            }
            for (int i = 1; i < numprocs; i++){
                MPI_Send(&r[0][0], N * 3, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }

            reduceBlock(numprocs);
            for (int i = 0; i < N; i++){
                for (int k = 0; k < 3; k++){
                    v[i][k] += 0.5 * a[i][k] * dt;
                }
            }
            file << instantaneousTemperature() << "\n";

        }
        file.close();
    	double t2 = MPI_Wtime();
	char tmp[256] = {0};
	sprintf(tmp,"mpi-%d.time",numprocs);
        ofstream file2(tmp);
        cout << "timespan:" << t2 - t1 << "s" << endl;
        file2 << t2 - t1 << endl;
        file2.close();
    }
    else{
        MPI_Status status;
        MPI_Recv(&r[0][0], N * 3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        long fromto[2] = { 0, 0 };
        MPI_Recv(&fromto[0], 2, MPI_LONG, 0, 2, MPI_COMM_WORLD, &status);
        computeAccelerations(myid, fromto[0], fromto[1]);
        MPI_Send(&a[0][0], N * 3, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

        for (int times = 0; times < TIMES; times++){
     
            MPI_Recv(&r[0][0], N * 3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            //MPI_Recv(&fromto[0], 2, MPI_LONG, 0, 2, MPI_COMM_WORLD, &status);
            computeAccelerations(myid, fromto[0], fromto[1]);
            MPI_Send(&a[0][0], N * 3, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}
