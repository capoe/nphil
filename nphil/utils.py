from . import maths
import nphil._nphil as _nphil
import numpy as np
import scipy.stats
import json

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

class RandomizeMatrix(object):
    def __init__(self, method):
        self.method = method
    def sample(self, X, n_samples, seed=None, log=None):
        rnd_X_list = []
        if seed != None: np.random.seed(seed)
        if self.method == "perm_within_cols":
            for i in range(n_samples):
                if log: log << log.back << "Random feature set" << i << log.flush
                rnd_X = np.copy(X)
                for col in range(X.shape[1]):
                    np.random.shuffle(rnd_X[:,col])
                rnd_X_list.append(rnd_X)
        elif self.method == "perm_rows":
            for i in range(n_samples):
                if log: log << log.back << "Random feature set" << i << log.flush
                rnd_X = np.copy(X)
                np.random.shuffle(rnd_X)
                rnd_X_list.append(rnd_X)
        else: raise ValueError(self.method)
        if log: log << log.endl
        return rnd_X_list

class CVLOO(object):
    def __init__(self, n_samples, options):
        self.tag = "cv_loo"
        self.n_samples = n_samples
        self.n_reps = n_samples
        self.step = 0
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_train = list(np.arange(self.step)) + list(np.arange(self.step+1, self.n_samples))
        idcs_test = [ self.step ]
        self.step += 1
        return info, idcs_train, idcs_test
    def isDone(self):
        return self.step >= self.n_reps

class CVMC(object):
    def __init__(self, n_samples, options):
        self.tag = "cv_mc"
        self.n_samples = n_samples
        self.n_reps = options.n_mccv
        self.f_mccv = options.f_mccv
        self.step = 0
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs = np.arange(self.n_samples)
        np.random.shuffle(idcs)
        split_at = int(self.f_mccv*self.n_samples)
        idcs_train = idcs[0:split_at]
        idcs_test = idcs[split_at:]
        self.step += 1
        return info, idcs_train, idcs_test
    def isDone(self):
        return self.step >= self.n_reps

class CVCustom(object):
    def __init__(self, n_samples, options):
        self.tag = "cv_custom"
        self.n_samples = n_samples
        self.splits = json.load(open(options.splits_json))
        self.n_reps = len(self.splits)
        self.step = 0
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_test = self.splits[self.step]["idcs_test"]
        mask = np.ones((self.n_samples,), dtype='i8')
        mask[idcs_test] = 0
        idcs_train = np.where(mask > 0)[0]
        idcs_test = np.where(mask == 0)[0]
        self.step += 1
        return info, idcs_train, idcs_test
    def isDone(self):
        return self.step >= self.n_reps

class CVUser(object):
    def __init__(self, n_samples, options):
        self.tag = "cv_user"
        self.n_samples = n_samples
        self.n_reps = 1
        self.step = 0
        self.mask = np.ones((self.n_samples,), dtype='i8')
        self.mask[options.test_on] = 0
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_train = np.where(self.mask > 0)[0]
        idcs_test = np.where(self.mask == 0)[0]
        self.step += 1
        return info, idcs_train, idcs_test
    def isDone(self):
        return self.step >= self.n_reps

class CVNone(object):
    def __init__(self, n_samples, options):
        self.tag = "cv_no"
        self.n_samples = n_samples
        self.n_reps = 1
        self.step = 0
    def next(self):
        assert not self.isDone()
        info = "%s_i%03d" % (self.tag, self.step)
        idcs_train = np.arange(self.n_samples)
        idcs_test = []
        self.step += 1
        return info, idcs_train, idcs_test
    def isDone(self):
        return self.step >= self.n_reps

def CVIter(tags, options):
    return cv_iterator[options.cv_mode](tags, options)

cv_iterator = {
  "loo": CVLOO,
  "mc": CVMC,
  "user": CVUser,
  "none": CVNone,
  "custom": CVCustom
}

def metric_mse(yp,yt):
    return np.sum((yp-yt)**2)/yp.shape[0]

def metric_rmse(yp,yt):
    return metric_mse(yp,yt)**0.5

def metric_mae(yp,yt):
    return np.sum(np.abs(yp-yt))/yp.shape[0]

def metric_rhop(yp,yt):
    return scipy.stats.pearsonr(yp, yt)[0]

def metric_rhor(yp,yt):
    return scipy.stats.spearmanr(yp, yt).correlation

def metric_auc(yp,yt):
    import sklearn.metrics
    return sklearn.metrics.roc_auc_score(yt,yp)

class CVEval(object):
    eval_map = { 
        "mae": metric_mae,
        "mse": metric_mse,
        "rmse": metric_rmse, 
        "rhop": metric_rhop,
        "auc":  metric_auc
    }
    def __init__(self, jsonfile=None):
        self.yp_map = {}
        self.yt_map = {}
        if jsonfile is not None: self.load(jsonfile)
        return
    def append(self, channel, yp, yt):
        if not channel in self.yp_map:
            self.yp_map[channel] = []
            self.yt_map[channel] = []
        self.yp_map[channel] = self.yp_map[channel] + list(yp)
        self.yt_map[channel] = self.yt_map[channel] + list(yt)
        return
    def evaluate(self, channel, metric, bootstrap=0):
        if len(self.yp_map[channel]) < 1: return np.nan
        if bootstrap == 0:
            return CVEval.eval_map[metric](
                np.array(self.yp_map[channel]), 
                np.array(self.yt_map[channel])), 0.
        else:
            v = []
            n = len(self.yp_map[channel])
            yp = np.array(self.yp_map[channel])
            yt = np.array(self.yt_map[channel])
            for r in range(bootstrap):
                re = np.random.randint(0, n, size=(n,))
                v.append(CVEval.eval_map[metric](yp[re], yt[re]))
            return np.mean(v), np.std(v)
    def evaluateNull(self, channel, metric, n_samples):
        if len(self.yp_map[channel]) < 1: return np.nan
        z = []
        for i in range(n_samples):
            yp_null = np.array(self.yp_map[channel])
            yt_null = np.array(self.yt_map[channel])
            np.random.shuffle(yp_null)
            z.append(CVEval.eval_map[metric](
                yp_null, yt_null))
        z = np.sort(np.array(z))
        return z
    def evaluateAll(self, metrics, bootstrap=0, log=None):
        res = {}
        for channel in sorted(self.yp_map):
            res[channel] = {}
            vs = []
            dvs = []
            for metric in metrics:
                v, dv = self.evaluate(channel, metric, bootstrap=bootstrap)
                res[channel][metric] = v
                res[channel][metric+"_std"] = dv
                vs.append(v)
                dvs.append(dv)
            if log:
                log << "%-9s : " % (channel) << log.flush
                for v, metric in zip(vs, metrics):
                    log << "%s=%+1.4e +- %+1.4e" % (
                        metric, v, dv) << log.flush
                log << log.endl
        return res
    def save(self, jsonfile):
        json.dump({ "yp_map": self.yp_map, "yt_map": self.yt_map },
            open(jsonfile, "w"), indent=1, sort_keys=True)
        return
    def load(self, jsonfile):
        data = json.load(open(jsonfile))
        self.yp_map = data["yp_map"]
        self.yt_map = data["yt_map"]
        return

class LSE(object):
    """Bootstrapper operating on user-specified prediction model

    Parameters
    ----------
    method: bootstrapping approach, can be 'samples', 'residuals' or 'features'
    bootstraps: number of bootstrap samples
    model: regressor/classifier object, e.g., sklearn.linear_model.LinearRegression,
        must implement fit and predict methods
    model_args: constructor arguments for model object
    """
    def __init__(self, **kwargs):
        self.method = kwargs["method"]
        self.bootstraps = kwargs["bootstraps"]
        self.model = kwargs["model"]
        self.model_args = kwargs["model_args"] if "model_args" in kwargs else {}
        self.ensemble = []
        self.feature_weights = None
    def fit(self, IX_train, Y_train, feature_weights=None):
        self.ensemble = []
        sample_iterator = resample_range(0, IX_train.shape[0], IX_train.shape[0])
        if self.method == 'samples':
            while len(self.ensemble) < self.bootstraps:
                resample_idcs = np.random.randint(IX_train.shape[0], size=(IX_train.shape[0],))
                m = self.model(**self.model_args)
                Y_train_boot = Y_train[resample_idcs]
                if np.std(Y_train_boot) < 1e-10: continue
                m.fit(IX_train[resample_idcs], Y_train[resample_idcs])
                self.ensemble.append(m)
        elif self.method == 'residuals':
            m = self.model(**self.model_args)
            m.fit(IX_train, Y_train)
            Y_train_pred = m.predict(IX_train)
            residuals = Y_train - Y_train_pred
            for bootidx in range(self.bootstraps):
                resample_idcs = np.random.randint(IX_train.shape[0], size=(IX_train.shape[0],))
                Y_train_resampled = Y_train + residuals[resample_idcs]
                m = self.model(**self.model_args)
                m.fit(IX_train, Y_train_resampled)
                self.ensemble.append(m)
        elif self.method == 'none':
            m = self.model(**self.model_args)
            m.fit(IX_train, Y_train)
            self.ensemble.append(m)
        elif self.method == 'features':
            if feature_weights is None: feature_weights = np.ones((IX_train.shape[1],))
            self.feature_weights = feature_weights
            self.feature_idcs = []
            n_features = IX_train.shape[1]
            weights = []
            for fidx in range(n_features):
                print("Ensemble for feature", fidx)
                if self.bootstraps > 0:
                    for bootidx in range(self.bootstraps):
                        resample_idcs = np.random.randint(IX_train.shape[0], size=(IX_train.shape[0],))
                        m = self.model(**self.model_args)
                        m.fit(IX_train[resample_idcs][:, [fidx]], Y_train[resample_idcs])
                        self.ensemble.append(m)
                        self.feature_idcs.append([fidx])
                        weights.append(feature_weights[fidx])
                else:
                    m = self.model(**self.model_args)
                    m.fit(IX_train[:,[fidx]], Y_train)
                    self.ensemble.append(m)
                    y = m.predict(IX_train[:,[fidx]])
                    self.feature_idcs.append([fidx])
                    weights.append(feature_weights[fidx])
            self.feature_weights = np.array(weights)
        else: raise ValueError(self.method)
    def predict(self, IX):
        Y_pred = []
        if self.method == 'features':
            for midx, m in enumerate(self.ensemble):
                Y_pred.append(m.predict(IX[:,self.feature_idcs[midx]]))
            Y_pred = np.array(Y_pred)
            Y_pred_med = []
            Y_pred_std = []
            for n in range(IX.shape[0]):
                y = Y_pred[:,n]
                order = np.argsort(y)
                y = y[order]
                w = self.feature_weights[order]
                w = np.cumsum(w)
                w = w/w[-1]
                s = np.searchsorted(w, 0.5)
                Y_pred_med.append(0.5*(y[s-1]+y[s]))
                Y_pred_std.append(np.std(y))
            avg = np.array(Y_pred_med)
            std = np.array(Y_pred_std)
        else:
            for m in self.ensemble:
                Y_pred.append(m.predict(IX))
            Y_pred = np.array(Y_pred)
            avg = np.median(Y_pred, axis=0)
            std = np.std(Y_pred, axis=0)
        return avg, std

class Booster(object):
    def __init__(self, options):
        self.options = options
        self.initialized = False
        # Cleared whenever dispatched with iter=0
        self.IX_trains = []
        self.Y_trains = []
        self.IX_tests = []
        self.Y_tests = []
        self.iteration = None
        # Kept across dispatches
        self.iteration_train_preds = {}
        self.iteration_train_trues = {}
        self.iteration_preds = {}
        self.iteration_trues = {}
        self.regressors = []
    def dispatchY(self, iteration, Y_train, Y_test):
        self.iteration = iteration
        if self.iteration == 0:
            self.IX_trains = []
            self.Y_trains = [ Y_train ]
            self.IX_tests = []
            self.Y_tests = [ Y_test ]
        if not self.iteration in self.iteration_preds:
            self.iteration_preds[self.iteration] = []
            self.iteration_trues[self.iteration] = []
            self.iteration_train_preds[self.iteration] = []
            self.iteration_train_trues[self.iteration] = []
    def dispatchX(self, iteration, IX_train, IX_test):
        assert iteration == self.iteration # Need to ::dispatchY first
        self.IX_trains.append(IX_train)
        self.IX_tests.append(IX_test)
    def getResidues(self):
        if self.iteration == 0:
            return self.Y_trains[0], self.Y_tests[0]
        else:
            return self.Y_trains[-1]-self.Y_trains[0], self.Y_tests[-1]-self.Y_tests[0]
    def train(self, regressor='lse', bootstraps=1000, method='samples', feature_weights=None, model_args={}):
        if type(regressor) == str and regressor == 'lse':
            import sklearn.linear_model
            model = sklearn.linear_model.LinearRegression
            regressor = LSE(bootstraps=bootstraps, method=method, model=model, model_args=model_args)
        elif type(regressor) == str and regressor == 'logit':
            import sklearn.linear_model
            model = sklearn.linear_model.LogisticRegression
            regressor = LSE(bootstraps=bootstraps, method=method, model=model, model_args=model_args)
        IX_train = np.concatenate(self.IX_trains, axis=1)
        Y_train = self.Y_trains[0]
        if feature_weights is None: regressor.fit(IX_train, Y_train)
        else: regressor.fit(IX_train, Y_train, feature_weights=feature_weights)
        self.regressors.append(regressor)
    def evaluate(self, method='moment'):
        IX_train = np.concatenate(self.IX_trains, axis=1)
        IX_test = np.concatenate(self.IX_tests, axis=1)
        Y_train = self.Y_trains[0]
        Y_test = self.Y_tests[0]
        Y_pred_train_avg, Y_pred_train_std = self.applyLatest(IX_train)
        Y_pred_test_avg, Y_pred_test_std = self.applyLatest(IX_test)
        # Log results
        self.iteration_train_preds[self.iteration].append(np.array([Y_pred_train_avg, Y_pred_train_std]).T)
        self.iteration_train_trues[self.iteration].append(Y_train.reshape((-1,1)))
        if IX_test.shape[0] > 0:
            self.iteration_preds[self.iteration].append(np.array([Y_pred_test_avg, Y_pred_test_std]).T)
            self.iteration_trues[self.iteration].append(Y_test.reshape((-1,1)))
        self.Y_trains.append(Y_pred_train_avg)
        self.Y_tests.append(Y_pred_test_avg)
        # Return stats
        if method == 'auroc':
            import sklearn.metrics
            auc_train = sklearn.metrics.roc_auc_score(Y_train, Y_pred_train_avg)
            mcc_train = sklearn.metrics.matthews_corrcoef(Y_train, Y_pred_train_avg)
            if IX_test.shape[0] > 1:
                auc_test = sklearn.metrics.roc_auc_score(Y_test, Y_pred_test_avg)
                mcc_test = sklearn.metrics.roc_auc_score(Y_test, Y_pred_test_avg)
            else:
                auc_test = np.nan
                mcc_test = np.nan
            return auc_train, mcc_train, auc_test, mcc_test
        else:
            import scipy.stats
            rmse_train = (np.sum((Y_pred_train_avg-Y_train)**2)/Y_train.shape[0])**0.5
            rho_train = scipy.stats.pearsonr(Y_pred_train_avg, Y_train)[0]
            if IX_test.shape[0] > 0:
                rmse_test = (np.sum((Y_pred_test_avg-Y_test)**2)/Y_test.shape[0])**0.5
                rho_test = scipy.stats.pearsonr(Y_pred_test_avg, Y_test)[0]
            else:
                rmse_test = np.nan
                rho_test = np.nan
            return rmse_train, rho_train, rmse_test, rho_test
    def applyLatest(self, IX):
        regressor = self.regressors[-1]
        Y_pred = []
        if IX.shape[0] > 0:
            Y_pred = regressor.predict(IX)
        if type(Y_pred) == tuple:
            Y_pred_avg = Y_pred[0]
            Y_pred_std = Y_pred[1]
        else:
            Y_pred_avg = Y_pred
            Y_pred_std = np.zeros(len(Y_pred))
        return Y_pred_avg, Y_pred_std
    def write(self, trunc='pred_i%d', log=None):
        iterations = sorted(self.iteration_preds)
        outfile_test = trunc+'_test.txt'
        outfile_train = trunc+'_train.txt'
        for it in iterations:
            # Training predictions
            preds_train = np.concatenate(self.iteration_train_preds[it], axis=0)
            trues_train = np.concatenate(self.iteration_train_trues[it], axis=0)
            np.savetxt(outfile_train % it, np.concatenate([preds_train, trues_train], axis=1))
            # Test predictions
            if len(self.iteration_preds[it]) > 0:
                preds_test = np.concatenate(self.iteration_preds[it], axis=0)
                trues_test = np.concatenate(self.iteration_trues[it], axis=0)
                np.savetxt(outfile_test % it, np.concatenate([preds_test, trues_test], axis=1))
            else:
                preds_test = []
                trues_test = []
                np.savetxt(outfile_test % it, np.array([]))
        return preds_train, trues_train, preds_test, trues_test

