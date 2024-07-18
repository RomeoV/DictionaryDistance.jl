module DictionaryDistance
import SparseArrays: nonzeroinds
import StatsBase: mean
using Hungarian, Distances
import MLBase: _roc

function align_dictionaries(D, D_approx)
    distances = [CosineDist()(v1, v2) for v1 in eachcol(D), v2 in eachcol(D_approx)]
    assignment, cost = hungarian(distances)
    (; assignment, cost)  # s.t. D_approx[:, assignment] ≈ D, and X[assignment, :] ≈ X
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
