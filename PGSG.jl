"""
Implementation of PGSG
"""
module PGSG

export pgsg, pf_pgsg, RPCA_PGSG, RPCA_PF_PGSG



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
    println("Inner loop finished")
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
        gamma = (t+1)^(beta)/10
        rho = 1/(2*gamma)
        x = stochasticGradient(x, grad_oracle, proj, gamma, rho, t, innerIterations(t, gamma, rho))
    end
    return x
end





function identity(x)
    return x
end

"Application to solving Robust PCA"
function RPCA_PGSG(U,V,A, gamma, rho, T)
    
    #### Computes a full subgradient, fairly slow since it uses the full dataset ####
    function deterministic_subgradient(X)
        U,V = X[1],X[2]
        R = U'*V
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            for j in 1:k #iterate over frames
                if R[i,j] - A[i,j]>0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                    end
                end
                if R[i,j] - A[i,j]<-0.001
                    for l in 1:n
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                    end
                end
            end
        end
        return [∇U, ∇V] 
    end

    #### Computes a subgradient from a random frame ####
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
        return [∇U, ∇V] 
    end

    #### Computes a subgradient from a random frame for a random component ####    
    function stochastic_subgradient2(X)
        U,V = X[1],X[2]
        R = U'*V
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            j = rand(1:k) #random over frames
                if R[i,j] - A[i,j]>0.001
                    l = rand(1:n) #random over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                end
                if R[i,j] - A[i,j]<-0.001
                    l = rand(1:n) #random over components
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                end
        end
        return [∇U, ∇V] 
    end

    return pf_pgsg([U,V], stochastic_subgradient2, identity, gamma, rho, T)
end


"Application to solving Robust PCA"
function RPCA_PF_PGSG(U,V,A, beta, T)
    #### Computes a full subgradient, fairly slow since it uses the full dataset ####
    function deterministic_subgradient(X)
        U,V = X[1],X[2]
        R = U'*V
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            for j in 1:k #iterate over frames
                if R[i,j] - A[i,j]>0.001
                    for l in 1:n #iterate over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                    end
                end
                if R[i,j] - A[i,j]<-0.001
                    for l in 1:n
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                    end
                end
            end
        end
        return [∇U, ∇V] 
    end

    #### Computes a subgradient from a random frame ####
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
        return [∇U, ∇V] 
    end

    #### Computes a subgradient from a random frame for a random component ####    
    function stochastic_subgradient2(X)
        U,V = X[1],X[2]
        R = U'*V
        
        n,m = size(U)
        n,k = size(V)
        ∇U = zeros(n,m)
        ∇V = zeros(n,k)
        
        for i in 1:m #iterate over pixels
            j = rand(1:k) #random over frames
                if R[i,j] - A[i,j]>0.001
                    l = rand(1:n) #random over components
                        ∇U[l,i] = ∇U[l,i] + V[l,j]
                        ∇V[l,j] = ∇V[l,j] + U[l,i]
                end
                if R[i,j] - A[i,j]<-0.001
                    l = rand(1:n) #random over components
                        ∇U[l,i] = ∇U[l,i] - V[l,j]
                        ∇V[l,j] = ∇V[l,j] - U[l,i]
                end
        end
        return [∇U, ∇V] 
    end

    return pf_pgsg([U,V], stochastic_subgradient2, identity, beta, T)
end


end #End of module PGSG