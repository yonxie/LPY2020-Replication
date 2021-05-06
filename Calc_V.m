function [v, d, phi, k, exps] = Calc_V(nz, neta, k, state, param)

% calculate firm value for grid z, eta, k with different policies. 

% Args: 
%       nz: number of the z grid
%       neta: num ber of eta grid
%       k: k state vector
%       state: state generated from DVI algorithm
%       param: parameter structer

% Return: 
%       v: value grid
%       d: dividend grid
%       phi: policy grid
%       k: k state vector
%       exps: expectation grid

G = -param.r + param.g_x + 0.5 * param.sigma_x^2 - param.sigma_x*param.lmda_e;
nk = length(k);

% calculate expectation 
[E_kt1s_0, E_kt1s_0_pure] = expectation(0, nz, neta, k, param.lmda_eta, param.delta, param.g_N, param.sigma_N, state);
[E_kt1s_1, E_kt1s_1_pure] = expectation(1, nz, neta, k, param.lmda_eta, param.delta, param.g_N, param.sigma_N, state);

for act = 0:1
    
    etas = permute(reshape(repmat(state.ETA, nz*nk, 1), [neta, nz, nk]), [2,1,3]);
    zs = reshape(repmat(state.Z, neta*nk, 1), [nz, neta, nk]);
    ks = permute(reshape(repmat(k, neta*nz, 1), [nk, neta, nz]), [3,2,1]);
    
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

v = vs0.*act0 + vs1.*act1;
phi = act1;
i_t = (exp(param.g_N+param.sigma_N*etas)./ks - 1 + param.delta) .* phi;
d = exp(zs) .* ks - (param.fa*ks + i_t.*ks) .* act1 - param.fo;
k = ks;

exps = E_kt1s_0_pure.*act0 + E_kt1s_1_pure.*act1;

end


                    
function [ex_vals_t, ex_vals_t_pure] = expectation(action, nz, neta, ks, lmda_eta, delta, g_N, sigma_N, state)

% Calculate expectation of E_t[exp(-lmda_eta.*eta_[t+1])*v_[t+1]]

% Return: expectation matrix at time t:nz*neta*nk

if action == 0
    k_t1 = (1-delta)*exp(-g_N-sigma_N * state.ETA)*ks'; 
else
    k_t1 = ones(length(state.ETA),length(ks));
end

vt1 = NaN(nz,neta,neta,length(ks));
vt1_pure = NaN(nz,neta,neta,length(ks));
for i = 1:nz
    for j=1:neta
        vt1_k = state.Vs(i,j,:);
        vt1(i,j,:,:) = exp(-lmda_eta*state.ETA(j)).*pchip(state.K,vt1_k,k_t1);
        vt1_pure(i,j,:,:) = pchip(state.K,vt1_k,k_t1);
    end
end

ex_vals_t = NaN(nz,neta,length(ks));
ex_vals_t_pure = NaN(nz,neta,length(ks));
for i = 1:nz
    probs=repmat((repmat((state.Z_PI(:,i)*state.ETA_P'),[1,1,neta])),[1,1,1,length(ks)]); %nz**neta*neta*nk
    ex_vals_t(i,:,:) = sum(sum(probs.*vt1,1),2);
    ex_vals_t_pure(i,:,:) = sum(sum(probs.*vt1_pure,1),2);
end

end