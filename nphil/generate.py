import nphil._nphil as _nphil
import numpy as np
import scipy.stats
import json
from . import maths
from .wrappers import *
from .utils import *
from .maths import CovTailScalingFct

def generate_graph(
        features_with_props,
        uop_list,
        bop_list,
        unit_min_exp,
        unit_max_exp,
        correlation_measure,
        rank_coeff=0.25):
    assert len(uop_list) == len(bop_list)
    fgraph = _nphil.FGraph(
        correlation_measure=correlation_measure, 
        unit_min_exp=unit_min_exp, 
        unit_max_exp=unit_max_exp,
        rank_coeff=rank_coeff)
    for f in features_with_props:
        fgraph.addRootNode(**f)
    for lidx in range(len(uop_list)):
        fgraph.addLayer(uop_list[lidx], bop_list[lidx])
    fgraph.generate()
    return fgraph

def infer_variable_properties(
        X_sample, 
        varnames=None, 
        units=None, 
        scales=None,
        zero=1e-5):
    n_vars = X_sample.shape[1]
    if varnames is None:
        varnames = [ "x%d" % _ for _ in range(n_vars) ]
    if units is None:
        units = [ "" ]*n_vars
    if scales is None:
        scales = [ 1. ]*n_vars
    variables = []
    for i in range(n_vars): 
        sample = X_sample[:,i]
        sample_min_abs = np.min(np.abs(sample))
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        if sample_min >= 0.: signstr = "+"
        elif sample_max <= 0.: signstr = "-"
        else: signstr = "+-"
        if signstr == "+-": zerostr = "+0"
        elif sample_min_abs > zero: zerostr = "-0"
        else: zerostr = "+0"
        variables.append({
            "varname": varnames[i], 
            "sign": signstr, 
            "zero": zerostr, 
            "scale": scales[i],
            "units": units[i] })
    return variables

def calculate_exceedence(
        covs_harm, 
        covs_sample, 
        epsilon=1e-10, 
        scale_fct=lambda c: c):
    # C = # channels
    # covs_harm: vec of length C
    # covs_sample: vec of length C
    # exs: vec of length C
    exs = (covs_sample-covs_harm)/(covs_harm)
    exs = exs*scale_fct(covs_sample)
    return exs

def fgraph_apply_batch(
        fgraph,
        IX_list,
        Y,
        log):
    if len(IX_list) > 0:
        npfga_dtype = IX_list[0].dtype
        covs = np.zeros((len(IX_list), len(fgraph)), dtype=npfga_dtype)
        IX_out = np.zeros((Y.shape[0],len(fgraph)), dtype=npfga_dtype)
        Y = Y.reshape((-1,1))
        for i, IX in enumerate(IX_list):
            log << log.back << "Randomized control, instance" << i << log.flush
            fgraph.applyAndCorrelate(IX, Y, IX_out, covs[i], Y.shape[0], Y.shape[1])
        log << log.endl
    else:
        covs = []
    return covs

def calculate_null_distribution(
        rand_covs,
        options,
        log,
        file_out=False):
    npfga_dtype = rand_covs.dtype
    # Dimensions and threshold
    n_channels = rand_covs.shape[1]
    n_samples = rand_covs.shape[0]
    p_threshold = 1. - options.tail_fraction
    i_threshold = int(p_threshold*n_samples+0.5)
    if log: log << "Tail contains %d samples" % (n_samples-i_threshold) << log.endl
    # Random-sampling convariance matrix
    # Rows -> sampling instances
    # Cols -> feature channels
    rand_cov_mat = np.copy(rand_covs)
    rand_cov_mat = np.abs(rand_cov_mat)
    # Sort covariance observations for each channel
    rand_covs = np.abs(rand_covs)
    rand_covs = np.sort(rand_covs, axis=0)
    # Fit scaling function
    cov_scaling_fct = CovTailScalingFct(rand_covs, options.tail_fraction)
    # Cumulative distribution for each channel
    rand_cum = np.ones((n_samples,1), dtype=npfga_dtype)
    rand_cum = np.cumsum(rand_cum, axis=0)
    rand_cum = (rand_cum-0.5) / rand_cum[-1,0]
    rand_cum = rand_cum[::-1,:]
    if file_out: np.savetxt('out_sis_channel_cov.hist', np.concatenate((rand_cum, rand_covs), axis=1))
    # Establish threshold for each channel
    thresholds = rand_covs[-int((1.-p_threshold)*n_samples),:]
    thresholds[np.where(thresholds < 1e-2)] = 1e-2
    t_min = np.min(thresholds)
    t_max = np.max(thresholds)
    t_std = np.std(thresholds)
    t_avg = np.average(thresholds)
    if log: log << "Channel-dependent thresholds: min avg max +/- std = %1.2f %1.2f %1.2f +/- %1.4f" % (
        t_min, t_avg, t_max, t_std) << log.endl
    # Peaks over threshold: calculate excesses for random samples
    if log: log << "Calculating excess for random samples" << log.endl
    pots = rand_covs[i_threshold:n_samples,:]
    pots = pots.shape[0]/np.sum(1./(pots+1e-10), axis=0) # harmonic average
    rand_exs_mat = np.zeros((n_samples,n_channels), dtype=npfga_dtype)
    for s in range(n_samples):
        if log: log << log.back << "- Sample %d/%d" % (s+1, n_samples) << log.flush
        rand_cov_sample = rand_cov_mat[s]
        exs = calculate_exceedence(pots, rand_cov_sample, scale_fct=cov_scaling_fct)
        rand_exs_mat[s,:] = exs
    # Random excess distributions
    rand_exs = np.sort(rand_exs_mat, axis=1) # n_samples x n_channels
    rand_exs_cum = np.ones((n_channels,1), dtype=npfga_dtype) # n_channels x 1
    rand_exs_cum = np.cumsum(rand_exs_cum, axis=0)
    rand_exs_cum = (rand_exs_cum-0.5) / rand_exs_cum[-1,0]
    rand_exs_cum = rand_exs_cum[::-1,:]
    rand_exs_avg = np.average(rand_exs, axis=0)
    rand_exs_std = np.std(rand_exs, axis=0)
    # Rank distributions: covariance
    rand_covs_rank = np.sort(rand_cov_mat, axis=1)
    rand_covs_rank = np.sort(rand_covs_rank, axis=0)
    rand_covs_rank = rand_covs_rank[:,::-1]
    # Rank distributions: exceedence
    rand_exs_rank = np.sort(rand_exs, axis=0) # n_samples x n_channels
    rand_exs_rank = rand_exs_rank[:,::-1]
    rand_exs_rank_cum = np.ones((n_samples,1), dtype=npfga_dtype) # n_samples x 1
    rand_exs_rank_cum = np.cumsum(rand_exs_rank_cum, axis=0)
    rand_exs_rank_cum = (rand_exs_rank_cum-0.5) / rand_exs_rank_cum[-1,0]
    rand_exs_rank_cum = rand_exs_rank_cum[::-1,:]
    if file_out: np.savetxt('out_exs_rank_rand.txt', np.concatenate([ rand_exs_rank_cum, rand_exs_rank ], axis=1))
    # ... Histogram
    if file_out: np.savetxt('out_exs_rand.txt', np.array([rand_exs_cum[:,0], rand_exs_avg, rand_exs_std]).T)
    if log: log << log.endl
    return pots, rand_exs_cum, rand_exs_rank_cum, rand_exs_rank, rand_covs_rank, rand_covs, cov_scaling_fct

def rank_ptest(
        tags,
        covs,
        exs,
        exs_cum,
        rand_exs_rank,
        rand_exs_rank_cum,
        file_out=False):
    n_channels = exs.shape[0]
    idcs_sorted = np.argsort(exs)[::-1]
    p_first_list = np.zeros((n_channels,))
    p_rank_list = np.zeros((n_channels,))
    for rank, c in enumerate(idcs_sorted):
        # Calculate probability to observe feature given its rank
        ii = np.searchsorted(rand_exs_rank[:,rank], exs[c])
        if ii >= rand_exs_rank_cum.shape[0]:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = 0.0
        elif ii <= 0:
            p0 = 1.0
            p1 = rand_exs_rank_cum[ii,0]
        else:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = rand_exs_rank_cum[ii,0]
        p_rank = 0.5*(p0+p1)
        # Calculate probability to observe feature as highest-ranked
        ii = np.searchsorted(rand_exs_rank[:,0], exs[c])
        if ii >= rand_exs_rank_cum.shape[0]:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = 0.0
        elif ii <= 0:
            p0 = 1.0
            p1 = rand_exs_rank_cum[ii,0]
        else:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = rand_exs_rank_cum[ii,0]
        p_first = 0.5*(p0+p1)
        p_first_list[c] = p_first
        p_rank_list[c] = p_rank
    if file_out:
        np.savetxt('out_exs_phys.txt', np.array([
            exs_cum[::-1,0],
            exs[idcs_sorted],
            p_rank_list[idcs_sorted],
            p_first_list[idcs_sorted]]).T)
    q_values = [ 1.-p_first_list[c] for c in range(n_channels) ]
    q_values_nth = [ 1.-p_rank_list[c] for c in range(n_channels) ]
    return q_values, q_values_nth

def resample_IX_Y(IX, Y, n):
    for i in range(n):
        idcs = np.random.randint(0, IX.shape[0], size=(IX.shape[0],))
        yield i, IX[idcs], Y[idcs]
    return

def mode_resample_IX_Y(IX, Y, n, threshold):
    idcs0 = np.where(Y < threshold)[0]
    idcs1 = np.where(Y >= threshold)[0]
    n0 = len(idcs0)
    n1 = len(idcs1)
    for i in range(n):
        re_idcs0 = idcs0[np.random.randint(0, n0, size=(n0,))]
        re_idcs1 = idcs1[np.random.randint(0, n1, size=(n1,))]
        idcs = list(re_idcs0) + list(re_idcs1)
        yield i, IX[idcs], Y[idcs]
    return

def resample_range(start, end, n):
    for i in range(n):
        idcs = np.random.randint(start, end, size=(end-start,))
        yield i, idcs
    return

def calculate_fnode_complexities(fgraph, op_complexity_map):
    root_dependencies = {}
    for fnode in fgraph:
        parents = fnode.getParents()
        c = op_complexity_map[fnode.op_tag] + sum([ f.q for f in parents ])
        fnode.q = c
        if fnode.is_root: 
            root_dependencies[fnode.expr] = { "cplx": fnode.q, "deps": { fnode.expr } }
        else:
            deps = set()
            for p in parents:
                deps = deps.union(root_dependencies[p.expr]["deps"])
            root_dependencies[fnode.expr] = { "cplx": fnode.q, "deps": deps }
    cplxs = [ f.q for f in fgraph ]
    return np.array(cplxs), sorted(set(cplxs))

def calculate_null_and_test(tags, covs, rand_covs, options, log, with_stats):
    pots_1xC, ranks_Cx1, ranks_Sx1, null_exs_SxC, null_order_covs_SxC, null_covs_SxC, cov_scaling_fct = calculate_null_distribution(
        rand_covs,
        options=options,
        log=log)
    # Test statistic: abs(cov)
    covs_abs = np.abs(covs)
    cq_values, cq_values_nth = rank_ptest(
        tags=tags,
        covs=covs_abs,
        exs=covs_abs,
        exs_cum=ranks_Cx1,
        rand_exs_rank=null_order_covs_SxC,
        rand_exs_rank_cum=ranks_Sx1)
    # Test statistic: exs(cov)
    exs = calculate_exceedence(pots_1xC, covs_abs, scale_fct=cov_scaling_fct)
    xq_values, xq_values_nth = rank_ptest(
        tags=tags,
        covs=covs_abs,
        exs=exs,
        exs_cum=ranks_Cx1,
        rand_exs_rank=null_exs_SxC,
        rand_exs_rank_cum=ranks_Sx1)
    if with_stats:
        cstats = PyFGraphStats(tags, covs, covs_abs, cq_values, cq_values_nth, null_order_covs_SxC, null_covs_SxC, cov_scaling_fct)
        xstats = PyFGraphStats(tags, covs, exs, xq_values, xq_values_nth, null_exs_SxC, null_covs_SxC, cov_scaling_fct)
    else:
        cstats = None
        xstats = None
    return null_order_covs_SxC, null_exs_SxC, covs_abs, exs, cstats, xstats

def run_npfga_with_phasing(fgraph, IX, Y, rand_IX_list, rand_Y, options, log):
    log << log.mg << "Running NPFGA with phasing" << log.endl
    # Hard-coded options
    edge_pctile = 100.
    null_edge_pctiles = [ 10. ]
    op_complexity_map = {
       "I": 0.0,
       "r": 0.75,
       "2": 0.75,
       "s": 1.00,
       "|": 1.25,
       "e": 1.50,
       "l": 1.50,
       "*": 1.75,
       ":": 2.00,
       "+": 2.25,
       "-": 2.25
    }
    # Precompute covariances across all channels
    fnodes_all = [ f for f in fgraph ]
    rand_covs_all = fgraph_apply_batch(fgraph, rand_IX_list, rand_Y, log)
    covs_all = fgraph.applyAndCorrelate(
        IX,
        Y.reshape((-1,1)),
        str(IX.dtype))[:,0]
    # Calculate complexities to inform phasing
    fnode_complexities, phase_thresholds = calculate_fnode_complexities(
        fgraph, op_complexity_map)
    phase_null_exs_top = []
    phase_null_cov_top = []
    phase_exs_top = []
    phase_cov_top = []
    phase_feature_idcs = []
    phase_cstats = []
    phase_xstats = []
    # Incrementally grow the active subgraph and evaluate
    for pidx, phase in enumerate(phase_thresholds):
        phase_idcs = np.where(fnode_complexities <= phase)[0]
        log << "Evaluating phase %2d: %5d nodes" % (pidx, len(phase_idcs)) << log.endl
        phase_feature_idcs.append(phase_idcs)
        null_order_covs_SxC, null_exs_SxC, covs, exs, cstats, xstats = calculate_null_and_test(
            tags=[ fnodes_all[_].expr for _ in phase_idcs ],
            covs=covs_all[phase_idcs],
            rand_covs=rand_covs_all[:, phase_idcs],
            options=options,
            log=None, #log,
            with_stats=True)
        phase_cstats.append(cstats)
        phase_xstats.append(xstats)
        # Store phased distributions
        covs_abs = np.abs(covs)
        phase_null_cov_top.append(null_order_covs_SxC[:,0])
        phase_cov_top.append(np.percentile(covs_abs, edge_pctile))
        phase_null_exs_top.append(null_exs_SxC[:,0])
        phase_exs_top.append(np.percentile(exs, edge_pctile))
    phase_null_exs_top = np.array(phase_null_exs_top).T # n_random_samples x n_phases
    phase_null_cov_top = np.array(phase_null_cov_top).T # n_random_samples x n_phases
    phase_exs_top = np.array(phase_exs_top).T # n_phases
    phase_cov_top = np.array(phase_cov_top).T # n_phases
    phase_offset_exs = np.zeros(phase_exs_top.shape, phase_exs_top.dtype)
    phase_offset_cov = np.zeros(phase_cov_top.shape, phase_cov_top.dtype)
    for pct in null_edge_pctiles:
        pct_null_exs = np.percentile(phase_null_exs_top, pct, axis=0)
        pct_null_cov = np.percentile(phase_null_cov_top, pct, axis=0)
        phase_offset_exs = phase_offset_exs + phase_exs_top - pct_null_exs
        phase_offset_cov = phase_offset_cov + phase_cov_top - pct_null_cov
    phase_offset_exs = phase_offset_exs/len(null_edge_pctiles)
    phase_offset_cov = phase_offset_cov/len(null_edge_pctiles)
    return phase_feature_idcs, phase_cstats, phase_xstats, phase_offset_cov, phase_offset_exs

def run_npfga(fgraph, IX, Y, rand_IX_list, rand_Y, options, log):
    """
    Required options fields: bootstrap, tail_fraction
    """
    # C = #channels, S = #samples
    rand_covs = fgraph_apply_batch(fgraph, rand_IX_list, rand_Y, log)
    # TODO For all phases ... >>>
    # TODO Too many return values, fix this
    pots_1xC, ranks_Cx1, ranks_Sx1, null_exs_SxC, \
        null_order_covs_SxC, null_covs_SxC, cov_scaling_fct \
            = calculate_null_distribution(
                rand_covs,
                options=options,
                log=log)
    # TODO <<< -> store
    # Bootstrap
    if options.bootstrap == 0:
        data_iterator = zip([0], [IX], [Y])
    elif options.bootstrap_by_mode:
        data_iterator = mode_resample_IX_Y(IX, Y, options.bootstrap, options.bootstrap_mode_threshold)
    else:
        data_iterator = resample_IX_Y(IX, Y, options.bootstrap)
    n_resamples = options.bootstrap if options.bootstrap > 0 else 1
    cov_samples    = np.zeros((len(fgraph),n_resamples), dtype=IX.dtype)
    exs_samples    = np.zeros((len(fgraph),n_resamples), dtype=IX.dtype)
    cq_samples     = np.zeros((len(fgraph),n_resamples), dtype=IX.dtype)
    cq_samples_nth = np.zeros((len(fgraph),n_resamples), dtype=IX.dtype)
    xq_samples     = np.zeros((len(fgraph),n_resamples), dtype=IX.dtype)
    xq_samples_nth = np.zeros((len(fgraph),n_resamples), dtype=IX.dtype)
    for sample_idx, IX_i, Y_i in data_iterator:
        if log: log << log.back << "Resampling idx" << sample_idx << log.flush
        Y_i = Y_i.reshape((-1,1))
        IX_i_out = np.zeros((Y_i.shape[0],len(fgraph)), dtype=IX_i.dtype)
        covs = np.zeros((len(fgraph),), dtype=IX_i.dtype)
        fgraph.applyAndCorrelate(
            IX_i,
            Y_i.reshape((-1,1)),
            IX_i_out, 
            covs,
            Y_i.shape[0],
            Y_i.shape[1])
        tags = [ f.expr for f in fgraph.getNodes() ]
        # TODO For all phases ... >>>
        # Test statistic: abs(cov)
        covs_abs = np.abs(covs)
        cq_values, cq_values_nth = rank_ptest(
            tags=tags,
            covs=covs_abs,
            exs=covs_abs,
            exs_cum=ranks_Cx1,
            rand_exs_rank=null_order_covs_SxC,
            rand_exs_rank_cum=ranks_Sx1)
        # Test statistic: exs(cov)
        exs = calculate_exceedence(pots_1xC, covs_abs, scale_fct=cov_scaling_fct)
        xq_values, xq_values_nth = rank_ptest(
            tags=tags,
            covs=covs_abs,
            exs=exs,
            exs_cum=ranks_Cx1,
            rand_exs_rank=null_exs_SxC,
            rand_exs_rank_cum=ranks_Sx1)
        cov_samples[:,sample_idx] = covs
        exs_samples[:,sample_idx] = exs
        cq_samples[:,sample_idx] = cq_values
        cq_samples_nth[:,sample_idx] = cq_values_nth
        xq_samples[:,sample_idx] = xq_values
        xq_samples_nth[:,sample_idx] = xq_values_nth
        # TODO <<< store
    if log: log << log.endl
    # Bootstrap avgs and stddevs
    covs = np.average(cov_samples, axis=1)
    covs_std = np.std(cov_samples, axis=1)
    exs = np.average(exs_samples, axis=1)
    exs_std = np.std(exs_samples, axis=1)
    cq_values = np.average(cq_samples, axis=1)
    cq_values_std = np.std(cq_samples, axis=1)
    xq_values = np.average(xq_samples, axis=1)
    xq_values_std = np.std(xq_samples, axis=1)
    cq_values_nth = np.average(cq_samples_nth, axis=1)
    cq_values_nth_std = np.std(cq_samples_nth, axis=1)
    xq_values_nth = np.average(xq_samples_nth, axis=1)
    xq_values_nth_std = np.std(xq_samples_nth, axis=1)
    for fidx, fnode in enumerate(fgraph.getNodes()):
        fnode.q = xq_values[fidx]
        fnode.cov = covs[fidx]
    cstats = PyFGraphStats(tags, covs, covs_abs, cq_values, cq_values_nth, null_order_covs_SxC, null_covs_SxC, cov_scaling_fct)
    xstats = PyFGraphStats(tags, covs, exs, xq_values, xq_values_nth, null_exs_SxC, null_covs_SxC, cov_scaling_fct)
    return tags, covs, covs_std, cq_values, cq_values_std, xq_values, xq_values_std, cstats, xstats

def solve_decomposition_lseq(input_tuples, bar_covs, log=None):
    # Setup linear system A*X = B and solve for x (margin terms)
    A = np.ones((len(input_tuples),len(input_tuples))) # coeff_matrix
    for i, row_tup in enumerate(input_tuples):
        for j, col_tup in enumerate(input_tuples):
            zero = False
            for tag in row_tup:
                if tag in col_tup:
                    zero = True
                    break
            if zero: A[i,j] = 0.0
    covs = np.zeros(shape=bar_covs.shape, dtype=bar_covs.dtype)
    if log: log << "Solving LSEQ" << log.endl
    for sample_idx in range(bar_covs.shape[2]):
        if log: log << log.back << " - Random sample %d" % (sample_idx) << log.flush
        covs[:,:,sample_idx] = np.linalg.solve(A, bar_covs[:,:,sample_idx])
    if log: log << log.endl
    return covs

def get_marginal_tuples(roots, fnodes, log=None):
    root_tags = [ r.expr for r in roots ]
    input_tuples = []
    max_size = max([ len(f.getRoots()) for f in fnodes ])
    if log: log << "Partial randomizations (max degree = %d)" % max_size << log.endl
    for size in range(len(root_tags)+1):
        if size > max_size: break
        tuples = maths.find_all_tuples_of_size(size, root_tags)
        if log: log << " - Degree %d: %d marginals" % (size, len(tuples)) << log.endl
        input_tuples.extend(tuples)
    return input_tuples

def run_cov_decomposition(fgraph, IX, Y, rand_IX_list, rand_Y, bootstrap, log=None):
    log << log.mg << "Nonlinear covariance decomposition" << log.endl
    roots = fgraph.getRoots()
    root_tag_to_idx = { r.expr: ridx for ridx, r in enumerate(roots) }
    input_tuples = get_marginal_tuples(roots, fgraph, log)
    # Calculate partially randomized marginals
    if bootstrap > 0:
        # Bootstrap sampling preps (bootstrap = 0 => no bootstrapping)
        n_resample = bootstrap if bootstrap > 0 else 1
        resample_iterator = resample_range(0, IX.shape[0], bootstrap) if bootstrap > 0 else zip([0], [ np.arange(0, IX.shape[0]) ])
        resample_iterator_reusable = [ r for r in resample_iterator ]
        bar_covs = np.zeros((len(input_tuples), len(fgraph), n_resample), dtype=IX.dtype)
        for tup_idx, tup in enumerate(input_tuples):
            log << "Marginal %d/%d: %s " % (tup_idx+1, len(input_tuples), tup) << log.endl
            rand_covs = np.zeros((len(fgraph), len(rand_IX_list), n_resample), dtype=IX.dtype)
            rand_Y = rand_Y.reshape((-1,1))
            for i in range(len(rand_IX_list)):
                rand_IX = np.copy(IX)
                for tag in tup:
                    rand_IX[:,root_tag_to_idx[tag]] = rand_IX_list[i][:,root_tag_to_idx[tag]]
                log << log.back << " - Randomized control, instance" << i << log.flush
                rand_IX_up = fgraph.apply(rand_IX, str(rand_IX.dtype))
                for boot_idx, idcs in resample_iterator_reusable:
                    y_norm = (Y[idcs]-np.average(Y[idcs]))/np.std(Y[idcs])
                    IX_up_norm, mean, std = maths.zscore(rand_IX_up[idcs])
                    rand_covs[:,i,boot_idx] = IX_up_norm.T.dot(y_norm)/y_norm.shape[0]
            log << log.endl
            bar_covs[tup_idx,:,:] = np.average(rand_covs, axis=1)
    else:
        bar_covs = np.zeros((len(input_tuples), len(fgraph), len(rand_IX_list)), dtype=IX.dtype)
        for tup_idx, tup in enumerate(input_tuples):
            log << "Marginal %d/%d: %s " % (tup_idx+1, len(input_tuples), tup) << log.endl
            rand_covs = np.zeros((len(fgraph), len(rand_IX_list)), dtype=IX.dtype)
            rand_Y = rand_Y.reshape((-1,1))
            for i in range(len(rand_IX_list)):
                rand_IX = np.copy(IX)
                for tag in tup:
                    rand_IX[:,root_tag_to_idx[tag]] = rand_IX_list[i][:,root_tag_to_idx[tag]]
                log << log.back << " - Randomized control, instance" << i << log.flush
                rand_covs[:,i] = fgraph.applyAndCorrelate(rand_IX, rand_Y, str(IX.dtype))[:,0]
            log << log.endl
            bar_covs[tup_idx,:,:] = rand_covs
    # Solve linear system for decomposition
    covs = solve_decomposition_lseq(input_tuples, bar_covs, log=log)
    covs_avg = np.average(covs, axis=2)
    covs_std = np.std(covs, axis=2)
    return input_tuples, covs_avg, covs_std

def run_cov_decomposition_filter(fgraph, order, IX, Y, rand_IX_list, rand_Y, bootstrap, log):
    fnodes = [ f for f in fgraph ]
    log << log.mg << "Cov decomposition filter" << log.endl
    keep = True
    selected_idx = None
    scores = []
    root_contributions_list = []
    for rank in xrange(-1,-len(order)-1,-1):
        fnode = fnodes[order[rank]]
        row_tuples, cov_decomposition, cov_decomposition_std = run_cov_decomposition_single(
            fgraph=fgraph,
            fnode=fnode,
            IX=IX,
            Y=Y,
            rand_IX_list=rand_IX_list,
            rand_Y=rand_Y,
            bootstrap=bootstrap,
            log=log)
        row_order = np.argsort(cov_decomposition[:,0])
        for r in row_order:
            log << "i...j = %-50s  cov(i..j) = %+1.4f (+-%1.4f)" % (
                row_tuples[r], cov_decomposition[r,0], cov_decomposition_std[r,0]) << log.endl
        root_tags = [ r.expr for r in fnode.getRoots() ]
        root_contributions = { t: { "cov": [], "std": [] } for t in root_tags }
        total_cov = np.sum(cov_decomposition)
        log << "Total covariance for this channel is" << total_cov << log.endl
        keep = True
        for r in root_tags:
            for row_idx, row_tuple in enumerate(row_tuples):
                if r in row_tuple:
                    root_contributions[r]["cov"].append(cov_decomposition[row_idx,0]/len(row_tuple))
                    root_contributions[r]["std"].append(cov_decomposition_std[row_idx,0])
            cov = np.array(root_contributions[r]["cov"])
            std = np.array(root_contributions[r]["std"])
            cov = np.sum(cov)
            std = (std.dot(std))**0.5
            root_contributions[r]["cov"] = cov
            root_contributions[r]["std"] = std
            if np.abs(cov) < 3.*std or cov*total_cov < 0.: # i.e., not significant or anticorrelated
                flag = 'x'
                keep = False
            else:
                flag = ''
            log << "x=%s => rho1(x) = %+1.4f +- %+1.4f    %s" % (r, cov, std, flag) << log.endl
        if keep:
            selected_idx = order[rank]
            #break
        delta = np.std([ root_contributions[r]["cov"] for r in root_tags ])
        log << "  => Score = |%1.4f| - %1.4f" % (total_cov, delta) << log.endl
        if np.isnan(delta):
            log << log.mr << "WARNING: NAN in covariance decomposition" << log.endl
        else:
            scores.append([ rank, np.abs(total_cov)-delta ])
        root_contributions_list.append([ fnode.expr, root_contributions])
    scores = sorted(scores, key=lambda s: -s[1])
    #if selected_idx is None: raise RuntimeError("Filter returned none")
    selected_idx = order[scores[0][0]]
    return selected_idx, root_contributions_list

def run_cov_decomposition_single(fgraph, fnode, IX, Y, rand_IX_list, rand_Y, bootstrap, log):
    log << log.mg << "Nonlinear covariance decomposition for '%s'" % fnode.expr << log.endl
    roots = fnode.getRoots()
    roots_all = fgraph.getRoots()
    root_tag_to_idx = { r.expr: ridx for ridx, r in enumerate(roots_all) }
    input_tuples = get_marginal_tuples(roots, [ fnode ], log)
    # Bootstrap sampling preps (bootstrap = 0 => no bootstrapping)
    n_resample = bootstrap if bootstrap > 0 else 1
    resample_iterator = resample_range(0, IX.shape[0], bootstrap) if bootstrap > 0 else zip([0], [ np.arange(0, IX.shape[0]) ])
    resample_iterator_reusable = [ r for r in resample_iterator ]
    # Calculate partially randomized marginals
    bar_covs = np.zeros((len(input_tuples), 1, n_resample), dtype=IX.dtype) # marginals x channels x resample
    for tup_idx, tup in enumerate(input_tuples):
        log << "Marginal %d/%d: %s " % (tup_idx+1, len(input_tuples), tup) << log.endl
        rand_covs = np.zeros((1, len(rand_IX_list), n_resample), dtype=IX.dtype) # channels x samples x resample
        rand_Y = rand_Y.reshape((-1,1))
        for i in range(len(rand_IX_list)):
            rand_IX = np.copy(IX)
            for tag in tup:
                rand_IX[:,root_tag_to_idx[tag]] = rand_IX_list[i][:,root_tag_to_idx[tag]]
            log << log.back << " - Randomized control, instance" << i << log.flush
            rand_x = fgraph.evaluateSingleNode(fnode, rand_IX, str(rand_IX.dtype))[:,0]
            for boot_idx, idcs in resample_iterator_reusable:
                y_norm = (Y[idcs]-np.average(Y[idcs]))/np.std(Y[idcs])
                x_norm = (rand_x[idcs] - np.average(rand_x[idcs]))/np.std(rand_x[idcs])
                rand_covs[0,i,boot_idx] = np.dot(x_norm, y_norm)/y_norm.shape[0]
        log << log.endl
        bar_covs[tup_idx,0,:] = np.average(rand_covs, axis=1)
    # Solve linear system for decomposition
    covs = solve_decomposition_lseq(input_tuples, bar_covs)
    covs_avg = np.average(covs, axis=2)
    covs_std = np.std(covs, axis=2)
    return input_tuples, covs_avg, covs_std

def calculate_root_weights(fgraph, q_values, row_tuples, cov_decomposition, log=None):
    if log: log << log.mg << "Calculating root weights from covariance decomposition" << log.endl
    root_tags = [ r.expr for r in fgraph.getRoots() ]
    row_idcs_non_null = list(filter(lambda i: len(row_tuples[i]) > 0, np.arange(len(row_tuples))))
    row_tuples_non_null = list(filter(lambda t: len(t) > 0, row_tuples))
    row_weights_non_null = [ 1./len(tup) for tup in row_tuples_non_null ]
    col_signs = np.sign(np.sum(cov_decomposition[row_idcs_non_null], axis=0))
    col_weights = np.array(q_values)
    tuple_weights = np.sum(cov_decomposition[row_idcs_non_null]*col_signs*col_weights, axis=1)
    root_counts = { root_tag: 0 for root_tag in root_tags }
    for f in fgraph:
        for r in f.getRoots(): root_counts[r.expr] += 1
    root_weights = { root_tag: 0 for root_tag in root_tags }
    for tupidx, tup in enumerate(row_tuples_non_null):
        for t in tup:
            root_weights[t] += 1./len(tup)*tuple_weights[tupidx]
    for r in root_weights: root_weights[r] /= root_counts[r]
    root_tags_sorted = sorted(root_tags, key=lambda t: root_weights[t])
    if log: log << "Tuple weight" << log.endl
    for tup_idx, tup in enumerate(row_tuples_non_null):
        if log: log << "i...j = %-50s  w(i...j) = %1.4e" % (':'.join(tup), tuple_weights[tup_idx]) << log.endl
    if log: log << "Aggregated root weight" << log.endl
    for r in root_tags_sorted:
        if log: log << "i = %-50s  w0(i) = %1.4e   (# derived nodes = %d)" % (r, root_weights[r], root_counts[r]) << log.endl
    return root_tags_sorted, root_weights, root_counts

def run_factor_analysis(mode, fgraph, fnode, IX, Y, rand_IX_list, rand_Y, ftag_to_idx, log):
    roots = fnode.getRoots()
    root_tags = [ (r.tag[2:-1] if r.tag.startswith("(-") else r.tag) for r in roots ]
    # Covariance for true instantiation
    x = fgraph.evaluateSingleNode(fnode, IX, str(IX.dtype))
    x_norm = (x[:,0]-np.average(x[:,0]))/np.std(x)
    y_norm = (Y-np.average(Y))/np.std(Y)
    cov = np.dot(x_norm, y_norm)/y_norm.shape[0]
    # Null dist
    rand_covs_base = []
    for i in range(len(rand_IX_list)):
        rand_x = fgraph.evaluateSingleNode(fnode, rand_IX_list[i], str(rand_IX_list[i].dtype))
        rand_x_norm = (rand_x[:,0] - np.average(rand_x[:,0]))/np.std(rand_x[:,0])
        rand_cov = np.dot(rand_x_norm, y_norm)/y_norm.shape[0]
        rand_covs_base.append(np.abs(rand_cov))
    rand_covs_base = np.array(sorted(rand_covs_base))
    np.savetxt('out_null.txt', np.array([np.arange(len(rand_covs_base))/float(len(rand_covs_base)), rand_covs_base]).T)
    # Analyse each factor
    factor_map = {}
    for root_tag in root_tags:
        rand_covs = []
        for i in range(len(rand_IX_list)):
            rand_IX = np.copy(IX)
            if mode == "randomize_this":
                rand_IX[:,ftag_to_idx[root_tag]] = rand_IX_list[i][:,ftag_to_idx[root_tag]]
            elif mode == "randomize_other":
                for tag in root_tags:
                    if tag == root_tag: pass
                    else: rand_IX[:,ftag_to_idx[tag]] = rand_IX_list[i][:,ftag_to_idx[tag]]
            else: raise ValueError(mode)
            rand_x = fgraph.evaluateSingleNode(fnode, rand_IX, str(rand_IX.dtype))
            rand_x_norm = (rand_x[:,0] - np.average(rand_x[:,0]))/np.std(rand_x[:,0])
            rand_cov = np.dot(rand_x_norm, y_norm)/y_norm.shape[0]
            rand_covs.append(np.abs(rand_cov))
        # Test
        rand_covs = np.array(sorted(rand_covs))
        np.savetxt('out_%s_%s.txt' % (mode.split("_")[1], root_tag), np.array([np.arange(len(rand_covs))/float(len(rand_covs)), rand_covs]).T)
        rank = np.searchsorted(rand_covs, np.abs(cov))
        q_value = float(rank)/len(rand_covs)
        factor_map[root_tag] = { "q_value": q_value, "min_cov": rand_covs[0], "max_cov": rand_covs[-1] }
    for factor, r in factor_map.iteritems():
        log << "%-50s c=%+1.4f q%-20s = %+1.4f  [random min=%1.2f max=%1.2f]" % (
            fnode.expr, cov, "(%s)" % factor, r["q_value"], r["min_cov"], r["max_cov"]) << log.endl
    return factor_map

