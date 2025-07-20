% Script for learning A using L1 loss
cvx_begin sdp
    variable A(n_reduced,n_reduced) symmetric
    minimize( norm(c0_multiplier*A(:)-c0_sample,1) )
    subject to
        A >= 0
cvx_end
a = A(:);