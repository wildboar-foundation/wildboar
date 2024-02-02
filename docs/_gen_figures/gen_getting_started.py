from light_dark import if_not_exists, yield_and_save_plot


@if_not_exists("getting-started", "interval.svg")
def gen_interval_explain(path, force=False):
    from wildboar.datasets import load_two_lead_ecg
    from wildboar.ensemble import ShapeletForestClassifier
    from wildboar.explain import IntervalImportance

    x, y = load_two_lead_ecg()

    clf = ShapeletForestClassifier(n_jobs=-1, n_shapelets=10, random_state=1)
    clf.fit(x, y)

    ex = IntervalImportance(window=8, random_state=1)
    ex.fit(clf, x, y)

    for fig, ax in yield_and_save_plot(path):
        _, mappable = ex.plot(x, y, ax=ax)
        fig.colorbar(mappable, ax=ax)
