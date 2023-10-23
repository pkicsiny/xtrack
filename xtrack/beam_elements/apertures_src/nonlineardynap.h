// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_NONLINEARDYNAP_H
#define XTRACK_NONLINEARDYNAP_H


/*gpufun*/
int64_t binary_search(const double* bins, int first, int last, const double x){
    // bins must be in increasing order: bins[i-1] > x >= bins[i]. Equivalent to np.digitize(right=False) or np.searchsorted(side='right')
    if (x >= bins[last])
        return last+1;

    while (first <= last)
    {
        int64_t middle = first + (last - first) / 2;
        if (x < bins[middle] && x >= bins[middle-1])
            return middle;
        if (bins[middle] <= x)
        {
            first = middle + 1;
        }
        else
            last = middle - 1;
    }
    return 0;
}


/*gpufun*/
void NonlinearDynap_track_local_particle(NonlinearDynapData el, LocalParticle* part0){
    FILE *f1 = fopen("/Users/pkicsiny/phd/cern/xsuite/notebooks_11_lxplus_papers/test.txt", "w");

    /*gpuglmem*/ double const* dynap_j_quadrant = NonlinearDynapData_getp1_dynap_j_quadrant(el, 0);  // 1d vector of len(delta_bin_edges)-1, J>=0, delta>=0, units of [nominal RMS sigma_j]
    /*gpuglmem*/ double const* delta_bin_edges  = NonlinearDynapData_getp1_delta_bin_edges(el, 0);  // delta>=0 bin edges, units of [nominal RMS sigma_delta]
    int64_t const num_bins = NonlinearDynapData_get_num_bins(el);  // len(delta_bin_edges)-1

    // nominal (unboosted) RMS beam sizes (e.g. from param table), not per slice!
    double const sigma_x     = NonlinearDynapData_get_sigma_x(el); 
    double const sigma_px    = NonlinearDynapData_get_sigma_px(el); 
    double const sigma_y     = NonlinearDynapData_get_sigma_y(el); 
    double const sigma_py    = NonlinearDynapData_get_sigma_py(el); 
    double const sigma_delta = NonlinearDynapData_get_sigma_delta(el); 

    //start_per_particle_block (part0->part)

        double const x  = LocalParticle_get_x (part);
        double const px = LocalParticle_get_px(part);
        double const y  = LocalParticle_get_y (part);
        double const py = LocalParticle_get_py(part);
        double const pzeta = LocalParticle_get_pzeta(part);

        // these I need to normalize by RMS sigmas
        double const jx = sqrt(x*x/(sigma_x*sigma_x) + px*px/(sigma_px*sigma_px)); // in untis of [sigma_j]
        double const jy = sqrt(y*y/(sigma_y*sigma_y) + py*py/(sigma_py*sigma_py)); // in untis of [sigma_j]
        double const pzeta_norm = pzeta/sigma_delta;  // in units of [sigma_delta]

         // find delta bin index
        int64_t delta_bin_idx = binary_search(delta_bin_edges, 0, num_bins, fabs(pzeta_norm));

        delta_bin_idx -= 1;  // 0 means before the first bin edge

        // select appropriate j limit [sigma_j]
        double j_max = dynap_j_quadrant[delta_bin_idx];
       
        fprintf(f1,"pzeta: %g, delta_bin_idx: %d, j_max: %g, (jx, jy): (%g, %g)\n", fabs(pzeta_norm), delta_bin_idx, j_max, jx, jy);


	int64_t const is_alive = (int64_t)(
		      (jx <= j_max) &&
		      (jy <= j_max) );

	// I assume that if I am in the function is because
    	if (!is_alive){
           LocalParticle_set_state(part, XT_LOST_ON_APERTURE);
	}

    //end_per_particle_block
    fclose(f1);
}
#endif
