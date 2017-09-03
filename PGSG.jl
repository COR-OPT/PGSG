"""
Implementation of PGSG
"""
module PGSG

export pgsg, pf_pgsg, RPCA_PGSG, RPCA_PF_PGSG, RPCA_subgradient



"Compute number of inner iterations to be done"
function innerIterations(t, gamma, rho)
    a = (4/(1-gamma*rho) +1)*(36/(1-gamma*rho))
    if 2*a*log(2*a) > 0
        return t + 1 #+ max(1, ceil(2*a*log(2*a)))
    else
        return t + 1
    end
end
"Compute step sizes"
function stepsize(t, j, gamma, rho)
    return 2*gamma/((1-gamma*rho)*(j+1) + 12/(1-gamma*rho))/17 #/2.65 *(t+1)^(-0.5)
end


"Run projected stochastic subgradient method on a strongly convex problem"
function stochasticGradient(y0, grad_oracle, proj, gamma, rho, t, numIterations)
    w = y0
    y = y0
    for j in 0:numIterations-2
        y = proj(y - stepsize(t,j,gamma,rho)*(grad_oracle(y) + (y-y0)/gamma))
        w = ((j+1)*w + y)/(j+2)
    end
    println("Inner loop finished ", norm(y0-w)/gamma)
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
function pf_pgsg(x0, grad_oracle, proj, beta, T)
    x = x0
    for t in 0:T-2
        rho = (t+1)^(beta)
        gamma = 1/(2*rho)
        x = stochasticGradient(x, grad_oracle, proj, gamma, rho, t, innerIterations(t, gamma, rho))
    end
    return x
end






"WITH OBJECTIVE TRACKING - Run projected stochastic subgradient method on a strongly convex problem"
function stochasticGradient(y0, grad_oracle, proj, gamma, rho, t, numIterations, inner_obj, objective, currentIterate)
    w = y0
    y = y0
    
    for j in 0:numIterations-2
        y = proj(y - stepsize(t,j,gamma,rho)*(grad_oracle(y) + (y-y0)/gamma))
        w = ((j+1)*w + y)/(j+2)
        inner_obj[currentIterate] = objective(w)
        currentIterate = currentIterate +1
    end
    println("Inner loop finished ", objective(w))
    return w, currentIterate
end
"WITH OBJECTIVE TRACKING - Run Parameter Free PGSG"
function pf_pgsg(x0, grad_oracle, proj, beta, T, objective)
    obj = zeros(convert(Int64, (T-1)*(T-2)/2))
    x = x0
    currentIterate = 1
    for t in 0:T-2
        rho = (t+1)^(beta)
        gamma = 1/(2*rho)
        x,currentIterate = stochasticGradient(x, grad_oracle, proj, gamma, rho, t, innerIterations(t, gamma, rho), obj, objective, currentIterate)
    end
    return x, obj
end
"WITH OBJECTIVE TRACKING - Run PGSG"
function pgsg(x0, grad_oracle, proj, gamma, rho, T, objective)
    obj = zeros(convert(Int64, (T-1)*(T-2)/2))
    x = x0
    currentIterate = 1
    for t in 0:T-2
        x,currentIterate = stochasticGradient(x, grad_oracle, proj, gamma, rho, t, innerIterations(t, gamma, rho), obj, objective, currentIterate)
    end
    return x, obj
end





function identity(x)
    return x
end

function oneNorm(A)
    r=0
    n,m = size(A)
    for i in 1:m
        for j in 1:m
            r = r + abs(A[i,j])
        end
    end
    return r
end
"Application to solving Robust PCA"
function RPCA_PGSG(U,V,A, gamma, rho, T, scale=1)

    #### Computes a subgradient from a random frame ####
    function stochastic_subgradient1(X)
        U,V = X[1],X[2]
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            j = rand(1:k) #random over frames
                if (U[:,i]'V[:,j])[1] - A[i,j]>0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                    end
                end
                if (U[:,i]'V[:,j])[1] - A[i,j]<-0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                    end
                end
        end
        return [∇U, ∇V] *scale
    end

    function objective_RPCA(X)
        return oneNorm(X[1].'*X[2] - A)
    end
    return pgsg([U,V], stochastic_subgradient1, identity, gamma, rho, T, objective_RPCA)
end


"Application to solving Robust PCA"
function RPCA_PF_PGSG(U,V,A, beta, T, scale=1)
    #### Computes a subgradient from a random frame ####
    function stochastic_subgradient1(X)
        U,V = X[1],X[2]
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            j = rand(1:k) #random over frames
                if (U[:,i]'V[:,j])[1] - A[i,j]>0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                    end
                end
                if (U[:,i]'V[:,j])[1] - A[i,j]<-0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                    end
                end
        end
        return [∇U, ∇V] *scale
    end

    function objective_RPCA(X)
        return oneNorm(X[1].'*X[2] - A)
    end
    return pf_pgsg([U,V], stochastic_subgradient1, identity, beta, T, objective_RPCA)
end


function RPCA_subgradient(U,V,A,T, scale=1)
    function stochastic_subgradient1(X)
        U,V = X[1],X[2]
        R = U'*V
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            j = rand(1:k) #random over frames
                if R[i,j] - A[i,j]>0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                    end
                end
                if R[i,j] - A[i,j]<-0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                    end
                end
        end
        return [∇U, ∇V] *scale
    end

    obj = zeros(T)
    alpha=0.01
    X = [U,V]
    for j in 1:T
        X = X - (alpha/sqrt(j))*stochastic_subgradient1(X)
        obj[j] = oneNorm(X[1].'*X[2] - A)
        if j%50==0 println("Inner loop finished ", oneNorm(X[1].'*X[2] - A)) end
    end
    return X, obj
end

end #End of module PGSG