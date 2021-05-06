
%% Description
% This script simulates the model in the paper 'the risk of old capital age
% : assge pricing implications of technology adoption'.

% notation in the paper is used in this script. 

%% Configurations

param = [];
% technology frontier
param.g_N = 0.02 / 4;           % Average log growth of the technology frontier
param.sigma_N = 0.0775;         % Volatility of log technology frontier

% aggregate productivity
param.g_x = 0.012/4;            % Average log growth of aggregate productivity
param.sigma_x = 0.047;          % Volatility of log aggregate productivity

% firm-specific productivity
param.rho_z = 0.97 ^ 3;         % Persistence of firm-specific productivity
param.z_bar = -2.8;             % Long run average of firm-specific productivity
param.sigma_z = 0.26;           % Conditional volatility of firm-specific productivity   

% technology capital
param.delta = 0.028;            % Rate of capital depreciation
param.fa = 0.8;                 % Fixed cost of technology adoption
param.fo = 0.012;               % fixed operating cost

% pricing kernel
param.r = 0.016/4;              % real risk-free rate
param.p_ex = 5;                 % Risk price of aggregate productivity shock
param.p_etaN = 8;               % Risk price of the technology frontier shock
param.lmda_e = param.p_ex*param.sigma_x;    % lambda e
param.lmda_eta = param.p_etaN*param.sigma_N;% lambda eta 

param.tol = 0.00001;             % value iteration tolerance
param.niter = 2000;             % num of iterations
param.k_low = -3;               % lower bound for log k
param.k_high = 0.5;               % upper bound for log k

% simulations
npnls = 10;                % number of panels
nfirms = 5000;            % number of firms
nqrtrs = 1120;            % number of quarters
nk = 2000;                % num of k grid
nz = 9;                   % num of z grid
neta = 9;                 % num of eta grid

% random seed
rng('default');
rng(42);


%% Panel Simulation

simulation = [];

for pnl = 1:npnls
    
    tic
    
    % calculate v function for each panel
    try
        state = load(['state_tol_' num2str(param.tol) '_kl_' num2str(param.k_low) '_kh_' num2str(param.k_high) '_nk_' num2str(nk) '_nz_' num2str(nz) '_neta_' num2str(neta) '.mat'], 'state');
        state = state.state;
    catch e
        tic
        state = DVI_V(nz, nk, neta, param);
        toc
        save(['state_tol_' num2str(param.tol) '_kl_' num2str(param.k_low) '_kh_' num2str(param.k_high) '_nk_' num2str(nk) '_nz_' num2str(nz) '_neta_' num2str(neta) '.mat'], 'state');
    end
   
    % calibrate splines
    state.phicps = [];
    for z = 1:nz
        for e = 1:neta
            state.phicps.(['phicp' num2str(z) num2str(e)]) = pchip(state.K, state.Vs(z, e, :));
        end
    end
    
    % sample etas: eta is same      
    etas_idx = randsample(1:neta, nqrtrs, true, state.ETA_P)';
    etas = state.ETA(etas_idx);
    
    % sample zs
    zs_idx = NaN(nqrtrs, nfirms);
    zs = NaN(nqrtrs, nfirms);
    for f = 1:nfirms
        zs_idx(:, f) = simulate(dtmc(state.Z_PI), nqrtrs-1);
        zs(:, f) = state.Z(zs_idx(:, f));
    end
    
    % calculate firm values, policies, and dividends
    vs = NaN(nqrtrs, nfirms); ds = NaN(nqrtrs, nfirms); 
    phis = NaN(nqrtrs, nfirms); ks = NaN(nqrtrs+1, nfirms);
    exps = NaN(nqrtrs, nfirms);
    ks(1,:) = 1;        % all firms start with k=1
    for q = 1:(nqrtrs)
        
        [v, d, phi, k, expe] = Calc_V(nz, neta, ks(q,:)', state, param);
               
        for f = 1:nfirms
            
           vs(q,f) = v(zs_idx(q,f), etas_idx(q), f);
           ds(q,f) = d(zs_idx(q,f), etas_idx(q), f);
           phis(q,f) = phi(zs_idx(q,f), etas_idx(q), f);
           exps(q,f) = expe(zs_idx(q,f), etas_idx(q), f);
           if phis(q,f) == 0
               ks(q+1,f) = (1-param.delta)*exp(-param.g_N-param.sigma_N*state.ETA(etas_idx(q)))*ks(q,f);
           else
               ks(q+1,f) = 1;
           end
           
        end
                
        disp(['finish panel:' num2str(pnl)  '; quarter: ' num2str(q)]);
        
    end
    
    % calculate returns
    rea_Rs = NaN(nqrtrs, nfirms); exp_Rs = NaN(nqrtrs, nfirms);
    rea_Rs(2:end,:) = vs(2:end,:) ./ (vs(1:(end-1),:) - ds(1:(end-1),:)) - 1;
    exp_Rs(1:end,:) = exps(1:end,:) ./ (vs(1:end,:) - ds(1:end,:)) - 1;
        
    simulations.(['panel' num2str(pnl)]).capital = ks;
    simulations.(['panel' num2str(pnl)]).value = vs;
    simulations.(['panel' num2str(pnl)]).dividend = ds;
    simulations.(['panel' num2str(pnl)]).adoption = phis;
    simulations.(['panel' num2str(pnl)]).z = zs;
    simulations.(['panel' num2str(pnl)]).eta = etas;
    simulations.(['panel' num2str(pnl)]).rea_return = rea_Rs;
    simulations.(['panel' num2str(pnl)]).exp_return = exp_Rs;
    
    save(['simulation_' 'npnls_' num2str(npnls) '_nfirms_' num2str(nfirms) '_tol_' num2str(param.tol) '_kl_' num2str(param.k_low) '_kh_' num2str(param.k_high) '_nk_' num2str(nk) '_nz_' num2str(nz) '_neta_' num2str(neta) '.mat'], 'simulations', '-v7.3');
    
    toc
    
end
