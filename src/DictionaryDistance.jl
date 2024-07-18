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
    # 
    idx_either = nonzeroinds(gt .|| pr)
    p = sum(gt)
    n = len - p
    tp = sum(gt .&& pr)
    tn = len - sum(gt .|| pr)
    fp = n - tn
    fn = p - tp
    return ROCNums{Int}(p, n, tp, tn, fp, fn)                                               
end

function compute_precision_recall_F_score(X_true, X_res; filter_smallest_quantile=false)
    scores = map(zip(eachcol(X_true), eachcol(X_res))) do (col_lhs, col_rhs)
        precision = sum(nonzeroinds(col_lhs) .∈ [nonzeroinds(col_rhs)]) / length(nonzeroinds(col_rhs))
        !isfinite(precision) && (precision = 0;)
        recall = sum(nonzeroinds(col_lhs) .∈ [nonzeroinds(col_rhs)]) / length(nonzeroinds(col_lhs))
        !isfinite(recall) && (recall = 0;)
        F_score = 2*(precision*recall) / max(precision + recall, 1)
        (; precision, recall, F_score)
    end
    (; precision=mean(getfield.(scores, :precision)),
       recall=mean(getfield.(scores, :recall)),
       F_score=mean(getfield.(scores, :F_score)),
     )
end
# Write your package code here.

# e.g. plot_nice_...(X, res.X[ass, :])
function plot_nice_side_by_side_view_of_sparse_results(X, X_res; filter_smallest_quantile=false)
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
