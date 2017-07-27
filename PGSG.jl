"""
Implementation of PGSG
"""
module PGSG

export pgsg



"Compute number of inner iterations to be done"
function innerIterations(t, gamma, rho)
    a = (4/(1-gamma*rho) +1)*(36/(1-gamma*rho))
    if 2*a*log(2*a) > 0
        return t + max(1, ceil(2*a*log(2*a)))
    else
        return t + 1
    end
end
"Compute step sizes"
function stepsize(t, j, gamma, rho)
    return 2*gamma/((1-gamma*rho)*(j+1) + 12/(1-gamma*rho))
end


"Run projected stochastic subgradient method on a strongly convex problem"
function stochasticGradient(y0, grad_oracle, proj, gamma, rho, t, numIterations)
    w = y0
    y = y0
    for j in 0:numIterations-2
        y = proj(y - stepsize(t,j,gamma,rho)*(grad_oracle(y) + (y-y0)/gamma))
        w = ((j+1)*w + y)/(j+2)
    end
    return w
end


"Run PGSG with user specified gamma"
function pgsg(x0, grad_oracle, proj, gamma, rho, T)
    x = x0
    for t in 0:T-2
        x = stochasticGradient(x, grad_oracle, proj, gamma, rho, t, innerIterations(t, gamma, rho))
    end
    return x
end

"Run Parameter Free PGSG"
function pf_pgsg(x0, grad_oracle, proj, T, beta)
    x = x0
    for t in 0:T-2
        gamma = pow(t+1, beta)
        rho = 1/(2*gamma)
        x = stochasticGradient(x, grad_oracle, proj, gamma, rho, t, innerIterations(t, gamma, rho))
    end
    return x
end

end #End of module PGSG