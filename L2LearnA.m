% Script for learning A using L2 loss
cvx_begin sdp
variable A(n_reduced,n_reduced) symmetric
minimize( ...
    (c0_multiplier*A(:)-c0_sample).'* ...
    (c0_multiplier*A(:)-c0_sample) )
subject to
    A >= 0
cvx_end
a = A(:);