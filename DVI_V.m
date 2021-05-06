function [state] = DVI_V(nz, nk, neta, param)


state = [];

% generate state z, k, eta
[state.Z, state.Z_PI] = rouwenhorst(nz, param.z_bar, param.rho_z, param.sigma_z);
state.ETA = [-3.1910 -2.2666 -1.4686 -0.7236 0 0.7236 1.4686 2.2666 3.1910]'; 
state.ETA_P = [0.00003961 0.0049436 0.08847453 0.43265156 0.7202352156 0.43265156 0.08847453 0.0049436 0.00003961]';        % column vector
state.ETA_P = state.ETA_P/sum(state.ETA_P);
state.K = exp(linspace(param.k_low, param.k_high, nk+1))';
state.K = state.K(2:end);

% common variables
G = -param.r + param.g_x + 0.5 * param.sigma_x^2 - param.sigma_x*param.lmda_e;

%% initialization
state.Vs = reshape(repmat(exp(state.Z) * state.K', neta, 1), [nz neta nk]);
disp(['finished initialiation.']);

%% value iteration
max_improvements = []; ave_improvements = []; iter = 1;

while 1

    % calculate expectation 
    [E_kt1s_0, E_kt1s_0_pure] = expectation(0, nz, neta, param.lmda_eta, param.delta, param.g_N, param.sigma_N, state);
    [E_kt1s_1, E_kt1s_1_pure] = expectation(1, nz, neta, param.lmda_eta, param.delta, param.g_N, param.sigma_N, state);

    % update values
    for act = 0:1

        % work on matrix of nz * neta * nk
        etas = permute(reshape(repmat(state.ETA, nz*nk, 1), [neta, nz, nk]), [2,1,3]);
        zs = reshape(repmat(state.Z, neta*nk, 1), [nz, neta, nk]);
        ks = permute(reshape(repmat(state.K, neta*nz, 1), [nk, neta, nz]), [3,2,1]);
        
        summand2 = exp(G - 0.5*param.lmda_eta^2+param.g_N+param.sigma_N*etas);    
        i_t = (exp(param.g_N+param.sigma_N*etas)./ks - 1 + param.delta) .* act;
        dt = exp(zs) .* ks - (param.fa*ks + i_t.*ks) .* act - param.fo;

        if act == 0
            vs0 = dt + summand2 .* E_kt1s_0;
        else
            vs1 = dt + summand2 .* E_kt1s_1;
        end

    end
  
    act0 = vs0 > vs1;
    act1 = vs0 <= vs1;
    new_Vs = vs0.*act0 + vs1.*act1;
        
    log_chge = abs(log(new_Vs./state.Vs));
    max_imprvmnt = max(log_chge, [], 'all');
    ave_imprvmnt = mean(log_chge, 'all');

    state.Vs = new_Vs;

    if max_imprvmnt < param.tol || iter > param.niter
        disp(['algorithm converged!']);
        state.phi = act1;
        state.trajectory = max_improvements;
        i_t = (exp(param.g_N+param.sigma_N*etas)./ks - 1 + param.delta) .* act1;
        state.d = exp(zs) .* ks - (param.fa*ks + i_t.*ks) .* act1 - param.fo;
        state.exp_Vs = act0.*E_kt1s_0_pure + act1.*E_kt1s_1_pure;
        break
    end
    
    max_improvements = [max_improvements max_imprvmnt];
    ave_improvements = [ave_improvements ave_imprvmnt];
    disp(['iter: ' num2str(iter) '; max change: ' num2str(max_imprvmnt) '; ave change: ' num2str(ave_imprvmnt)]);
    
    iter = iter + 1;
    
end        

plot(max_improvements);
plot(ave_improvements);

end


%% Functions

function [ex_vals_t, ex_vals_t_pure] = expectation(action, nz, neta, lmda_eta, delta, g_N, sigma_N, state)
% Calculate expectation of E_t[exp(-lmda_eta.*eta_[t+1])*v_[t+1]]
%Return: expectation matrix at time t:nz*neta*nk

if action == 0
    k_t1 = (1-delta)*exp(-g_N-sigma_N * state.ETA)*state.K'; 
else
    k_t1 = ones(length(state.ETA),length(state.K));
end

vt1 = NaN(nz,neta,neta,length(state.K));
vt1_pure = NaN(nz,neta,neta,length(state.K));
for i = 1:nz
    for j=1:neta
        vt1_k = state.Vs(i,j,:);
        vt1(i,j,:,:) = exp(-lmda_eta*state.ETA(j)).*pchip(state.K,vt1_k,k_t1);
        vt1_pure(i,j,:,:) = pchip(state.K,vt1_k,k_t1);
    end
end

ex_vals_t = NaN(nz,neta,length(state.K));
ex_vals_t_pure = NaN(nz,neta,length(state.K));
for i = 1:nz
    probs=repmat((repmat((state.Z_PI(:,i)*state.ETA_P'),[1,1,neta])),[1,1,1,length(state.K)]); %nz**neta*neta*nk
    ex_vals_t(i,:,:) = sum(sum(probs.*vt1,1),2);
    ex_vals_t_pure(i,:,:) = sum(sum(probs.*vt1_pure,1),2);
end

end
