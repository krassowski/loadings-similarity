from math import sqrt
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from numpy import Inf
from tqdm.auto import tqdm
from pandas import DataFrame


def threshold_as_is(threshold, **kargs):
    return threshold


def threshold_adaptive(components_importance_measure: str, combine=True):

    assert components_importance_measure in {'individual', 'cumulative'}

    def calc_threshold(threshold, train_pca, test_pca, n, m):

        if components_importance_measure == 'individual':
            importance = (
                sqrt(train_pca.explained_variance_ratio_[n - 1] * test_pca.explained_variance_ratio_[m - 1])
                if combine else
                train_pca.explained_variance_ratio_[n - 1]
            )
            return 1 - importance
        else:
            cummulative_importance = sum(train_pca.explained_variance_ratio_[:n])
            if combine:
                cummulative_importance *= sum(test_pca.explained_variance_ratio_[:m])
                cummulative_importance = sqrt(cummulative_importance)
            return cummulative_importance
        assert False

    return calc_threshold


def loadings_similarity(
    pipeline: Pipeline, data, limit_to_n_components=Inf, cv=KFold(), expectation_correction=False,
    distance_correction=False,
    threshold=0.5, threshold_modifier=None
):
    result = []
    keeps = []

    train_pipeline = clone(pipeline)
    test_pipeline = clone(pipeline)

    if not threshold_modifier:
        threshold_modifier = threshold_as_is

    for train, test in tqdm(cv.split(data), total=cv.get_n_splits()):

        train = data[train]
        test = data[test]

        train_pipeline.fit(train)
        train_pca = train_pipeline.steps[-1][1]
        train_loadings = train_pca.components_

        test_pipeline.fit(test)
        test_pca = test_pipeline.steps[-1][1]
        test_loadings = test_pca.components_

        components_n = min(limit_to_n_components, len(train_loadings))

        if expectation_correction:
            expected_hits = {}
            # test_loading = test_loadings[m - 1]
            # correlation, pvalue = method(test_loading, trained_loading)
            # assert isclose(correlation, correlations[n-1, m-1])
            for n in range(1, components_n + 1):
                loading_a = train_loadings[n - 1]
                self_hits = 0
                for m in range(1, components_n + 1):
                    loading_b = train_loadings[m - 1]
                    correlation, pvalue = pearsonr(loading_a, loading_b)
                    if abs(correlation) >= threshold:
                        self_hits += 1

                expected_hits[n] = self_hits

        cv_keeps = []
        correlations = 1 - cdist(train_loadings, test_loadings, 'correlation')

        for n in range(1, components_n + 1):

            abs_correlations = []
            correlations_adj = []

            corresponding_pc_in_test = None
            keep = False

            test_components_n = min(len(test_loadings), components_n)
            for m in range(1, test_components_n + 1):

                corrected_correlation = 1 - abs(correlations[n - 1, m - 1])
                adjusted_threshold = threshold_modifier(
                    threshold,
                    train_pca=train_pca, test_pca=test_pca,
                    n=n, m=m
                )

                abs_correlations.append(corrected_correlation)

                if expectation_correction:
                    corrected_correlation = corrected_correlation * expected_hits[n]

                if distance_correction:
                    distance = 1 + abs(n - m)
                    adjusted_threshold = 1 - (
                        (1 - threshold)
                        *
                        (1 - abs(n/components_n - m/test_components_n))
                    )
                    corrected_correlation = corrected_correlation * distance

                correlations_adj.append(corrected_correlation)

                if corrected_correlation < 1 - adjusted_threshold:
                    keep = True
                    corresponding_pc_in_test = m
                    break

            cv_keeps.append(keep)

            result.append({
                'n': n,
                'best_abs_corr': 1 - min(abs_correlations),
                'best_abs_corr_adj': 1 - min(correlations_adj),
                'threshold': threshold,
                'adjusted_threshold': adjusted_threshold,
                'corresponding_pc_in_test': corresponding_pc_in_test,
                'keep': keep,
                'expected': expected_hits[n] if expectation_correction else None
            })

        keeps.extend([sum(cv_keeps)] * components_n)

    df = DataFrame(result)
    df['keep_n'] = keeps
    return df
