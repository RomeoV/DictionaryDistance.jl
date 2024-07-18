"""
    module DictionaryDistance

Help evaluate how well a problem ``Y \\approx D * X`` has been solved.
This module offers two main functions:
1. `align_dictionaries`, to help match found dictionaries `D` to the original dictionaries, and
2. `roc`, with additional functions `precision`, `recall`, and `f1score`.

Also, the helper function `nonzerovec` is included.

A typical workflow might look as follows:

```julia
D = ...
X = ...
Y = D*X
res = ksvd(Y, size(D, 2))

D_est, X_est = res.D, res.X
(; assignment) = align_dictionaries(D, D_est)
X_est_perm = X_est_perm[assignment, :]

r = roc(nonzerovec(X[:]), nonzerovec(X_est_perm[:]))
@show precision(r), recall(r), f1score(r)
```
"""
module DictionaryDistance
import SparseArrays: nonzeroinds, SparseVector, nnz, sparse
using Hungarian, Distances
import MLBase: _roc, ROCNums
import Reexport: @reexport
@reexport import MLBase: roc, precision, recall, f1score

export align_dictionaries, nonzerovec

"""
    align_dictionaries(D_lhs, D_rhs)

Finds the best assignment of dictionary vectors of `D_rhs` and `D_lhs`.
Specifically, this function returns assignments s.t. D_rhs[:, assignment] ≈ D_lhs.
If this comes frome some setup `Y ≈ D * X` then X_rhs[assignment, :] ≈ X_lhs.
"""
function align_dictionaries(D_lhs::AbstractMatrix, D_rhs::AbstractMatrix)
    distances = [CosineDist()(v1, v2) for v1 in eachcol(D_lhs), v2 in eachcol(D_rhs)]
    assignment, cost = hungarian(distances)
    (; assignment, cost)  # s.t. D_rhs[:, assignment] ≈ D_lhs, and X_rhs[assignment, :] ≈ X_lhs
end

# function precition
nonzerovec(vec::SparseVector{<:Number}) = SparseVector(length(vec), nonzeroinds(vec), fill(true, nnz(vec)))

function _roc(gt::SparseVector{Bool}, pr::SparseVector{Bool})
    len = length(gt)                                                                        
    length(pr) == len || throw(DimensionMismatch("Inconsistent lengths."))                  
    p = sum(gt)
    n = len - p
    tp = sum(gt .&& pr)
    tn = len - sum(gt .|| pr)
    fp = n - tn
    fn = p - tp
    return ROCNums{Int}(p, n, tp, tn, fp, fn)                                               
end

# e.g. plot_nice_...(X, res.X[ass, :])
function _plot_nice_side_by_side_view_of_sparse_results(X, X_res; filter_smallest_quantile=false)
    if filter_smallest_quantile
        X[X .< 0.25] .= 0
        X_res[X_res .< 0.25] .= 0
    end
    col_idx = 1:20
    row_idx = findall((sum.(!iszero, eachrow(X[:,col_idx])) .!= 0) .|| (sum.(!iszero, eachrow(X_res[:,col_idx])) .!= 0))
    S = (length(row_idx), 1)
    visual_column = [sparse(zeros(S)),sparse(ones(S)),sparse(zeros(S))]
    cat(([X[row_idx, i] X_res[row_idx, i] visual_column...] for i in col_idx)...;
        dims=2)
end


end
