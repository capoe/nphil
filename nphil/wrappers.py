from . import maths
import nphil._nphil as _nphil
import numpy as np
import scipy.stats
import json

class PyFGraph(object):
    def __init__(self, fgraph):
        self.fgraph_ = fgraph
        self.fnodes = []
        self.fnode_map = {}
        self.generations = None
        self.root_nodes = None
        self.map_generations = None
        self.extract()
        self.rank()
    def extract(self):
        self.fnodes = [ PyFNode(_) for _ in self.fgraph_ ]
        self.root_nodes = list(filter(lambda f: f.is_root, self.fnodes))
        self.map_generations = {}
        self.generations = sorted(list(set([ f.generation for f in self.fnodes ])))
        self.map_generations = { g: [] for g in self.generations }
        for f in self.fnodes:
            self.map_generations[f.generation].append(f)
        print("FGraph of size %d" % len(self.fnodes))
        for g in self.generations:
            print("  %4d nodes at generation %d" % (
                len(self.map_generations[g]), g))
        self.fnode_map = { f.expr: f for f in self.fnodes }
        for f in self.fnodes: f.resolveParents(self.fnode_map)
        return
    def rank(self, cumulative=False, ordinal=False, key=lambda f: np.abs(f.cov*f.confidence)):
        scores = [ key(f) for f in self.fnodes ]
        scores_cum = np.cumsum(sorted(scores))
        ranked = sorted(self.fnodes, key=key)
        for idx, r in enumerate(ranked):
            if ordinal:
                r.rank = float(idx)/(len(ranked)-1.)
            elif cumulative == True:
                r.rank = scores_cum[idx]/scores_cum[-1]*key(r)
            else:
                r.rank = key(r)
        return

class PyFGraphStats(object):
    def __init__(self, 
            tags, 
            covs, 
            exs, 
            q_values, 
            q_values_nth, 
            null_exs, 
            null_covs, 
            cov_tail_scaling_fct):
        self.n_samples = null_exs.shape[0]
        self.n_channels = null_exs.shape[1]
        self.tags = tags
        self.covs = covs
        self.exs = exs
        self.q_values = np.array(q_values)
        self.q_values_nth = np.array(q_values_nth)
        self.null_exs = null_exs # matrix of size SxC (samples by channels)
        self.null_covs = null_covs # matrix of size SxC
        self.cov_tail_scaling_fct = cov_tail_scaling_fct
        self.order = np.argsort(-np.abs(self.covs))
        self.evaluateTopNode()
    def evaluateTopNode(self):
        self.top_idx = self.order[0]
        self.top_tag = self.tags[self.top_idx]
        self.top_cov = self.covs[self.top_idx]
        self.top_q = self.q_values[self.top_idx]
        self.top_exceedence = self.exs[self.top_idx]
        self.top_avg_null_exceedence = np.average(self.null_exs[:,0])
        self.top_rho_harm = np.abs(self.top_cov)/(1.+self.top_exceedence)
        self.top_avg_null_cov = self.top_cov*(1.+self.top_avg_null_exceedence)/(1.+self.top_exceedence)
        self.percentiles = np.arange(0,110,10)
        self.top_avg_null_exc_percentiles = [ np.percentile(self.null_exs[:,0], p) for p in self.percentiles ]
        self.top_avg_null_cov_percentiles = [ self.top_cov*(1.+e)/(1.+self.top_exceedence) for e in self.top_avg_null_exc_percentiles ]
        return
    def calculateCovExceedencePercentiles(self, pctiles=None, log=None):
        if pctiles is None: pctiles = np.arange(0,110,10)
        covx_1st_list = []
        covx_nth_list = []
        for r in range(self.n_channels):
            covx_1st = self.calculateExpectedCovExceedenceRank(rank=r, rank_null=0, pctiles=pctiles)
            covx_nth = self.calculateExpectedCovExceedenceRank(rank=r, rank_null=r, pctiles=pctiles)
            covx_1st_list.append(covx_1st)
            covx_nth_list.append(covx_nth)
        covx_1st = np.array(covx_1st_list).T
        covx_nth = np.array(covx_nth_list).T
        return self.order, pctiles, covx_1st, covx_nth
    def calculateExpectedCovExceedenceRank(self, rank, rank_null, pctiles):
        idx = self.order[rank]
        cov = np.abs(self.covs[idx])
        exc = self.exs[idx]
        null_exceedence_pctiles = np.array([ np.percentile(self.null_exs[:,rank_null], p) for p in pctiles ])
        rho_harm = cov/(1.+exc) # harmonic tail cov for this channel
        null_cov_pctiles = cov*(1.+null_exceedence_pctiles)/(1.+exc)
        return -null_cov_pctiles+cov
    def summarize(self, log):
        log << "Top-ranked node: '%s'" % (self.top_tag) << log.endl
        log << "  [phys]   cov  = %+1.4f     exc  = %+1.4f     q = %1.4f" % (self.top_cov, self.top_exceedence, self.top_q) << log.endl
        log << "  [null]  <cov> = %+1.4f    <exc> = %+1.4f" % (self.top_avg_null_cov, self.top_avg_null_exceedence) << log.endl
        log << "Percentiles"
        for idx, p in enumerate(self.percentiles):
            log << "  [null] p = %1.2f  <cov>_p = %+1.4f  <exc>_p = %+1.4f" % (
                0.01*p, self.top_avg_null_cov_percentiles[idx], self.top_avg_null_exc_percentiles[idx]) << log.endl
        cidx = np.argmax(np.abs(self.covs))
        eidx = np.argmax(self.exs)
        qidx = np.argmax(self.q_values)
        log << "Max cov observed: c=%+1.4f @ %s" % (self.covs[cidx], self.tags[cidx]) << log.endl
        log << "Max exc observed: e=%+1.4f @ %s" % (self.exs[eidx], self.tags[eidx]) << log.endl
        log << "Max prb observed: q=%+1.4f @ %s" % (self.q_values[qidx], self.tags[qidx]) << log.endl
        return
    def tabulateExceedence(self, outfile):
        percentiles = np.arange(0, 110, 10)
        null_exs_percentiles = []
        for p in percentiles:
            null_exs_percentiles.append(np.percentile(self.null_exs, p, axis=0))
        null_exs_percentiles = np.array(null_exs_percentiles).T
        ranks = np.arange(len(self.exs))+1.
        ranks = ranks/ranks[-1]
        ofs = open(outfile, 'w')
        ofs.write('# rank exs null@' + ' null@'.join(list(map(str, percentiles))) + '\n')
        chunk = np.concatenate([ ranks.reshape((-1,1)), np.sort(self.exs)[::-1].reshape((-1,1)), null_exs_percentiles ], axis=1)
        np.savetxt(ofs, chunk)
        ofs.close()
        return
    def getChannelNullCovDist(self, channel_idx, ofs=None):
        dist = np.array([ 1.-np.arange(self.n_samples)/float(self.n_samples), self.null_covs[:,channel_idx]]).T
        if ofs: np.savetxt(ofs, dist)
        return dist

class PyFNode(object):
    def __init__(self, fnode):
        self.fnode_ = fnode
        self.parents_ = self.fnode_.getParents()
        self.parents = []
        self.is_root = fnode.is_root
        self.generation = fnode.generation
        self.expr = fnode.expr
        self.cov = fnode.cov
        self.confidence = fnode.q
        self.rank = -1
    def resolveParents(self, fnode_map):
        self.parents = [ fnode_map[p.expr] for p in self.parents_ ]

