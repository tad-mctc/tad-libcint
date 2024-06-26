/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <math.h>
#define MIN(X,Y)        ((X)<(Y) ? (X) : (Y))
#define MAX(X,Y)        ((X)>(Y) ? (X) : (Y))

/*
 * def condense(op, a, loc):
 *     nd = loc[-1]
 *     out = numpy.empty((nd,nd))
 *     for i,i0 in enumerate(loc):
 *         i1 = loc[i+1]
 *         for j,j0 in enumerate(loc):
 *             j1 = loc[j+1]
 *             out[i,j] = op(a[i0:i1,j0:j1])
 *     return out
 */

void NPcondense(double (*op)(double *, int, int, int), double *out, double *a,
                int *loc_x, int *loc_y, int nloc_x, int nloc_y)
{
        const size_t nj = loc_y[nloc_y];
        const size_t Nloc_y = nloc_y;
#pragma omp parallel
{
        int i, j, i0, j0, di, dj;
#pragma omp for
        for (i = 0; i < nloc_x; i++) {
                i0 = loc_x[i];
                di = loc_x[i+1] - i0;
                for (j = 0; j < nloc_y; j++) {
                        j0 = loc_y[j];
                        dj = loc_y[j+1] - j0;
                        out[i*Nloc_y+j] = op(a+i0*nj+j0, nj, di, dj);
                }
        }
}
}

double NP_sum(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = 0;
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out += a[i*nd+j];
        } }
        return out;
}
double NP_max(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = a[0];
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out = MAX(out, a[i*nd+j]);
        } }
        return out;
}
double NP_min(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = a[0];
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out = MIN(out, a[i*nd+j]);
        } }
        return out;
}
double NP_abssum(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = 0;
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out += fabs(a[i*nd+j]);
        } }
        return out;
}
double NP_absmax(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = fabs(a[0]);
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out = MAX(out, fabs(a[i*nd+j]));
        } }
        return out;
}
double NP_absmin(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = fabs(a[0]);
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out = MIN(out, fabs(a[i*nd+j]));
        } }
        return out;
}
double NP_norm(double *a, int nd, int di, int dj)
{
        int i, j;
        double out = 0;
        for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
                out += a[i*nd+j] * a[i*nd+j];
        } }
        return sqrt(out);
}
