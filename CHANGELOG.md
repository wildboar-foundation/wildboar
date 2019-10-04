# 0.3.1

## Major features
 
No major features
 
## Minor features

### Improved caching of lower-bound for DTW

The DTW subsequence search implementation has been improved by caching
DTW lower-bound information for repeated calls with the same
subsequece. This slightly increases the memory requirement, but can
give significantly improved performance under certain circumstances.
 
### Allow shapelet information to be extracted
 
A new attribute `ts_info` is added to `Shapelet` (which is accessible
from `tree.root_node_.shapelet`). `ts_info` returns a tuple
`(ts_index, ts_start, length)` with information about the index (in
the `x` used to fit, `fit(x, y)`, the model) and the start position of
the shapelet. For a shapelet tree/forest fit on `x` the shapelet in a
particular node is given by `x[ts_index, ts_start:(ts_start +
length)]`.

## Bug 
 
No bug fixes
 
# 0.3

## Major features

### Regression shapelet trees

A new type of shapelet trees has been
added. `wildboar.tree.ShapeletTreeRegressor` which allows for
constructing shapelet trees used to predict real value outputs.

### Regression shapelet forest

A new tyoe of shapelet forest has been
added. `wildboar.ensemble.ShapeletForestRegressor` which allows for
constructing shapelet forests for predicting real value outputs.

## Minor features

No minor features

## Bug fixes

 * a6f656d Fix bug for strided labels not correctly accounted for
 * 321a04d Remove unused property `unscaled_threshold`
 * 134ec77 Fix bug #1
 * ad2fc2c Fix issue #2
