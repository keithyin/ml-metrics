import numpy as np
import sklearn.metrics as sklearn_metrics


class BiClassificationMetrics(object):

    def Reset(self):
        raise NotImplementedError()

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        raise NotImplementedError()

    def Compute(self, to_string=False):
        raise NotImplementedError()


class Mape(BiClassificationMetrics):
    def __init__(self, down_sample_rate=1., name="ctr"):
        """
        :param down_sample_rate: float or None
            negative-sample down sample rate in training stage.
            if you down sample the inference dataset too, no need to set the down sample rate
        :return:
        """
        self._name = name
        self._down_sample_rate = down_sample_rate
        self._cur_num = 0
        self._accu_mape = 0
        self._accu_mape_raw = 0
        pass

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.flatten()
        labels[labels < 1e-8] = 1e-8
        predicts = predicts.flatten()
        assert labels.shape[0] == predicts.shape[0]
        self._cur_num += labels.shape[0]

        self._accu_mape_raw += np.sum(np.abs(labels - predicts) / labels)

        # correction
        predicts = predicts / (predicts + (1-predicts)/self._down_sample_rate)
        self._accu_mape += np.sum(np.abs(labels - predicts) / labels)

    def Compute(self, to_string=False):
        """
        :return: [mape_raw, mape]   mape_raw: uncorrected, mape: corrected
        """
        if self._cur_num == 0:
            raise ValueError("Run Update first")

        # before correction
        mape_raw = self._accu_mape_raw / self._cur_num
        mape = self._accu_mape / self._cur_num
        results = [mape_raw, mape]
        if to_string:
            results = "{}--> mape_raw:{:.5f}, mape:{:.5f}, ".format(self._name, *results)
        return results


class MAE(BiClassificationMetrics):
    def __init__(self):
        self._tot_loss = 0
        self._tot_num = 0

    def Reset(self):
        self._tot_loss = 0
        self._tot_num = 0

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        labels = labels.flatten()
        predicts = predicts.flatten()
        assert labels.shape[0] == predicts.shape[0]
        self._tot_num += labels.shape[0]
        self._tot_loss = np.sum(np.abs(labels - predicts))

    def Compute(self, to_string=False):
        result = self._tot_loss / self._tot_num
        if to_string:
            result = "mae:{:.5f}, ".format(result)
        return result


class Copc(BiClassificationMetrics):
    def __init__(self, down_sample_rate=1., name="ctr"):
        """
        :param down_sample_rate: float or None. used for ctr correction
            negative-sample down sample rate in training stage.
            if you down sample the inference dataset too, no need to set the down sample rate
        :return:
        """
        self._name = name
        self._down_sample_rate = down_sample_rate
        self._cur_num = 0
        self._accumulated_labels = 0.
        self._accumulated_predicts = 0.
        self._accumulated_predicts_raw = 0.
        pass

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.flatten()
        predicts = predicts.flatten()

        assert labels.shape[0] == predicts.shape[0]
        self._accumulated_predicts_raw += np.sum(predicts)
        predicts = predicts / (predicts + (1-predicts)/self._down_sample_rate)
        self._cur_num += labels.shape[0]
        self._accumulated_labels += np.sum(labels)
        self._accumulated_predicts += np.sum(predicts)

    def Compute(self, to_string=False):
        if self._cur_num == 0:
            raise ValueError("Run Update first")
        real_ctr = self._accumulated_labels / self._cur_num
        p_ctr = self._accumulated_predicts / self._cur_num
        p_ctr_raw = self._accumulated_predicts_raw / self._cur_num
        copc_val = p_ctr / real_ctr - 1.
        copc_raw = p_ctr_raw / real_ctr - 1.
        results = real_ctr, p_ctr_raw, copc_raw, p_ctr, copc_val
        if to_string:
            results = "{}--> real:{:.3%}, predicted_raw:{:.3%}, copc_raw:{:.3%}, predicted:{:.3%}, copc:{:.3%}, ".format(
                self._name, *results)
        return results


class Gcopc(BiClassificationMetrics):
    def __init__(self, down_sample_rate=1., name="ctr"):
        """
        :param down_sample_rate: float or None. used for ctr correction
            negative-sample down sample rate in training stage.
            if you down sample the inference dataset too, no need to set the down sample rate
        :return:
        """
        self._name = name
        self._down_sample_rate = down_sample_rate
        self._key_counter = {}
        self._copcs = {}

    def Reset(self):
        self._down_sample_rate = 1.0
        self._key_counter = {}
        self._copcs = {}

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :param kwargs: kwargs["keys"] is list
        :return: None
        """
        labels = labels.flatten()
        predicts = predicts.flatten()
        assert "keys" in kwargs
        keys = kwargs["keys"]
        assert labels.shape[0] == predicts.shape[0]
        assert len(keys) == labels.shape[0]

        for i, key in enumerate(keys):
            if key not in self._copcs:
                self._copcs[key] = Copc(down_sample_rate=self._down_sample_rate, name=self._name)
                self._key_counter[key] = 0
            self._key_counter[key] += 1
            self._copcs[key].Update(labels[i: i + 1], predicts[i: i + 1])

    def Verbose(self):
        res_str = "\ngcopc_verbose\n"
        res_infos = []
        for k, copc in self._copcs.items():
            tmp_str = "key:{}, {}".format(k, copc.Compute(to_string=True))
            res_infos.append([k, tmp_str])

        res_infos = sorted(res_infos, key=lambda x: x[0])
        res_infos = [info[1] for info in res_infos]
        res_infos = "\n".join(res_infos) + "\n"
        res_str += res_infos
        return res_str

    def Compute(self, to_string=False):
        tot = sum(self._key_counter.values())
        gcopc = 0.
        gcopc_raw = 0.
        g_real_ctr = 0.
        g_p_raw_ctr = 0.
        g_p_ctr = 0.

        for k, count in self._key_counter.items():
            ratio = count / tot
            r_ctr, p_ctr_raw, copc_raw, p_ctr, copc_val = self._copcs[k].Compute()
            g_real_ctr += (ratio * r_ctr)
            g_p_raw_ctr += (ratio * p_ctr_raw)
            gcopc_raw += (ratio * copc_raw)
            gcopc += (ratio * copc_val)
            g_p_ctr += (ratio * p_ctr)

        results = g_real_ctr, g_p_raw_ctr, gcopc_raw, g_p_ctr, gcopc
        if to_string:  # raw mean before correction
            results = "{}--> g_real:{:.3%}, g_p_raw:{:.3%}, g_copc_raw:{:.3%}, g_predicted:{:.3%}, g_copc:{:.3%}, ".format(
                self._name, *results)
        return results


class Auc(BiClassificationMetrics):
    def __init__(self, num_buckets, name="ctr"):
        self._name = name
        self._num_buckets = num_buckets
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Reset(self):
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.flatten()
        predicts = predicts.flatten()

        assert len(labels.shape) == 1
        assert len(predicts.shape) == 1

        labels = labels.astype(np.int)
        predicts = self._num_buckets * predicts

        buckets = np.round(predicts).astype(np.int)
        buckets = np.where(buckets < self._num_buckets,
                           buckets, self._num_buckets-1)

        for i in range(len(labels)):
            self._table[labels[i], buckets[i]] += 1

    def Compute(self, to_string=False):
        tn = 0
        tp = 0
        area = 0
        for i in range(self._num_buckets):
            new_tn = tn + self._table[0, i]
            new_tp = tp + self._table[1, i]
            # self._table[1, i] * tn + self._table[1, i]*self._table[0, i] / 2
            area += (new_tp - tp) * (tn + new_tn) / 2
            tn = new_tn
            tp = new_tp
        if tp < 1e-3 or tn < 1e-3:
            return -0.5  # 样本全正例，或全负例
        results = area / (tn * tp)
        if to_string:
            results = "{}--> auc:{:.5f}, ".format(self._name, results)
        return results


class Gauc(BiClassificationMetrics):
    def __init__(self, num_buckets=10240, name="ctr"):
        self._name = name
        self._aucs = {}
        self._key_counter = {}
        self._num_buckets = num_buckets

    def Update(self, labels: np.ndarray, predicts: np.ndarray, **kwargs):
        """
        :param labels:
        :param predicts:
        :param kwargs: must keys, kwargs['keys'] list
        :return:
        """
        labels = labels.flatten()
        predicts = predicts.flatten()
        assert "keys" in kwargs
        keys = kwargs['keys']
        assert labels.shape[0] == predicts.shape[0]
        assert len(keys) == labels.shape[0]

        for i, key in enumerate(keys):
            if key not in self._key_counter:
                self._key_counter[key] = 0
                self._aucs[key] = Auc(num_buckets=self._num_buckets, name=self._name)
            self._key_counter[key] += 1
            self._aucs[key].Update(labels[i: i+1], predicts[i: i+1])

    def Verbose(self):
        tot = sum(self._key_counter.values())
        res_str = "\ngauc_verbose\n"
        res_infos = []
        for k, count in self._key_counter.items():
            proportion = count / tot
            tmp_str = "key:{}, proportion:{:.3%}, auc:{:.5f}".format(k, proportion, self._aucs[k].Compute())
            res_infos.append([k, tmp_str])  # just for sort
        res_infos = sorted(res_infos, key=lambda x: x[0])
        res_infos = [info[1] for info in res_infos]
        res_infos = "\n".join(res_infos) + "\n"
        res_str += res_infos
        return res_str

    def Compute(self, to_string=False):
        tot = sum(self._key_counter.values())
        gauc = 0.
        for k, count in self._key_counter.items():
            gauc += (count / tot * self._aucs[k].Compute())
        if to_string:
            gauc = "{}--> gauc:{:.5f}, ".format(self._name, gauc)
        return gauc


if __name__ == '__main__':
    label = np.random.randint(low=0, high=2, size=[10000])
    predict = np.random.uniform(0, 1, size=[10000])
    auc = Auc(num_buckets=1024)
    auc.Update(label, predict)
    gauc = Gauc(10240)
    gauc.Update(label, predict, keys=list(np.ones(shape=[10000], dtype=np.int)))
    print(auc.Compute(True))
    print(gauc.Verbose())
    print(sklearn_metrics.roc_auc_score(label, predict))

    copc = Copc(down_sample_rate=0.1)
    ones = (1 / 11) * np.ones(shape=[100])

    act_ones = np.ones(shape=[100])
    copc.Update(ones, ones)
    print(copc.Compute())

    gcopc = Gcopc(down_sample_rate=0.1, name="cvr")
    gcopc.Update(labels=ones, predicts=ones, keys=list(np.ones(shape=[100], dtype=np.int).flatten()))
    gcopc.Update(labels=ones, predicts=ones, keys=list(2*np.ones(shape=[100], dtype=np.int).flatten()))

    print(gcopc.Compute(to_string=True))
    print(gcopc.Verbose())
