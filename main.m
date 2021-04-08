
%% Description
% This script simulates the model in the paper 'the risk of old capital age
% : assge pricing implications of technology adoption'.

% notation in the paper is used in this script. 

global state probs

%% Configurations


% technology frontier
g_N = 0.02 / 4;           % Average log growth of the technology frontier
sigma_N = 0.0775;         % Volatility of log technology frontier

% aggregate productivity
g_x = 0.012/4;            % Average log growth of aggregate productivity
sigma_x = 0.047;          % Volatility of log aggregate productivity

% firm-specific productivity
rho_z = 0.97 ^ 3;         % Persistence of firm-specific productivity
z_bar = -2.8;             % Long run average of firm-specific productivity
sigma_z = 0.26;           % Conditional volatility of firm-specific productivity   

% technology capital
delta = 0.028;            % Rate of capital depreciation
fa = 0.8;                 % Fixed cost of technology adoption
fo = 0.012;               % fixed operating cost

% pricing kernel
r = 0.016/4;              % real risk-free rate
p_ex = 5;                 % Risk price of aggregate productivity shock
p_etaN = 8;               % Risk price of the technology frontier shock
lmda_e = p_ex*sigma_x;    % lambda e
lmda_eta = p_etaN*sigma_N;% lambda eta 


% simulations
npnls = 1;                % number of panels
nfirms = 5000;            % number of firms
nqrtrs = 1120;            % number of quarters
tol = 0.0001;             % value iteration tolerance
nk = 1000;                % num of k grid
nz = 9;                  % num of z grid
neta = 9;                 % num of eta grid

% random seed
rng('default');
rng(42);

%% Simulate V        
state = [];

% generate state z, k, eta
[state.Z, state.Z_PI] = rouwenhorst(nz, z_bar, rho_z, sigma_z);
state.ETA = [-3.1910 -2.2666 -1.4686 -0.7236 0 0.7236 1.4686 2.2666 3.1910]'; 
state.ETA_P = [0.00003961 0.0049436 0.08847453 0.43265156 0.7202352156 0.43265156 0.08847453 0.0049436 0.00003961]';        % column vector
state.ETA_P = state.ETA_P/sum(state.ETA_P);
state.K = exp(linspace(-2, 2, nk+1))';
state.K = state.K(2:end);

% common variables
G = -r + g_x + 0.5 * sigma_x^2 - sigma_x*lmda_e;

%% initialization
state.Vs = reshape(repmat(exp(state.Z) * state.K', neta, 1), [nz neta nk]);
probs = transit_probs(neta, nz, state);
disp(['finished initialiation.']);

%% value iteration
improvements = []; iter = 1; niter = 500000;

while 1

    % calculate expectation 
    E_kt1s_0 = expectation(0, nz, neta, lmda_eta, delta, g_N, sigma_N);
    E_kt1s_1 = expectation(1, nz, neta, lmda_eta, delta, g_N, sigma_N);

    % update values
    for act = 0:1

        % work on matrix of nz * neta * nk
        etas = reshape(repmat(state.ETA, nz*nk, 1), [nz, neta, nk]);
        zs = reshape(repmat(state.Z, neta*nk, 1), [nz, neta, nk]);
        ks = reshape(repmat(state.K, neta*nz, 1), [nz, neta, nk]);

        summand2 = exp(G - 0.5*lmda_eta^2+g_N+sigma_N*etas);    
        i_t = (exp(g_N+sigma_N*etas)./ks - 1 + delta) .* act;
        dt = exp(zs) .* ks - (fa*ks + i_t.*ks) .* act - fo;

        if act == 0
            vs0 = dt + summand2 .* E_kt1s_0;
        else
            vs1 = dt + summand2 .* E_kt1s_1;
        end

    end

    act0 = vs0 > vs1;
    act1 = vs0 <= vs1;
    new_Vs = vs0.*act0 + vs1.*act1;

    imprvmnt = max(abs(log(new_Vs./state.Vs)), [], 'all');

    state.Vs = new_Vs;

    if imprvmnt < tol || iter > niter
        disp(['algorithm converged!'])
        break
    end
    
    improvements = [improvements imprvmnt];
    disp(['iter: ' num2str(iter) '; improvement: ' num2str(imprvmnt)]);
    
    iter = iter + 1;
    
end        

plot(improvements);

%% Functions
function [ex_vals_t] = expectation(action, nz, neta, lmda_eta, delta, g_N, sigma_N)

% Calculate expectation of E_t[exp(-lambda_eta, * eta_{t+1}) * v_{t+1}]

% Args:
%   nz: number of z
%   action: 
%   neta: number of eta
%   other parameters


% Return: expectation matrix at time t: nz * neta * nk

global state probs

if action == 0
    k_t1 = (1-delta)*exp(-g_N-sigma_N*state.ETA)*state.K';   % k_{t+1}: neta * nk
else
    k_t1 = ones(length(state.ETA), length(state.K));         % k_{t+1}: neta * nk
end

vt1 = NaN(neta, nz, neta, length(state.K));

for i = 1:nz
    for j = 1:neta
        vt1_k = squeeze(state.Vs(i, j, :));
        vt1(j,i,:,:) = exp(-lmda_eta*state.ETA(j)) .* pchip(state.K, vt1_k, k_t1);
    end
end

ex_vals_t = NaN(nz, neta,length(state.K));
for i = 1:nz
    prob = squeeze(probs(i,:,:,:,:));
    ex_vals_t(i,:,:) = sum(sum(prob.*vt1, 1), 2);
end

end

function [probs] = transit_probs(neta, nz, state)
    
    probs = NaN(nz, neta, nz, neta, length(state.K));
    
    for i = 1:nz
        prob = state.ETA_P * state.Z_PI(i,:);
        for jj = 1:neta
            for mm = 1:length(state.K)
                probs(i, :, :, jj, mm) = prob;
            end
        end
    end
              
end
