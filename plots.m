
state = load('state_tol_1e-05_kl_-3_kh_0.5_nk_2000_nz_9_neta_9.mat');
% simu = load('simulation_npnls_1_nfirms_5000_tol_1e-05_kl_-3_kh_0.5_nk_2000_nk_2000_nz_9_neta_9.mat', 'simulations');
% simu = simu.simulations.panel1;
% state = load('state.mat', 'state');

state = state.state;

nz = 9; neta = 9; nk = 2000; nfirms=5000;
r = 0.016/4;

%% optimal policy & firm value

% ks = reshape(permute(reshape(repmat(state.K, neta*nz, 1), [nk, neta, nz]), [3,2,1]), [nz*neta*nk, 1]);
% vs = reshape(state.Vs, [nz*neta*nk, 1]);
% ds = reshape(state.d, [nz*neta*nk, 1]);

% 2D plot
ld = 2;

ks = state.K;
vs = squeeze(state.Vs(7,5,:));
ds = squeeze(state.d(7,5,:));
phis = squeeze(state.phi(7,5,:));
expvs = squeeze(state.exp_Vs(7,5,:));
exprs = expvs./(vs - ds) - 1;

non_apt = phis==0;
subplot(2,2,1);

% plot(ks(non_apt), phis(non_apt), 15, '.', 'b')
plot(ks(non_apt), phis(non_apt), 'linewidth', ld)
xlim([0 1.7])
hold on 
% plot(ks(~non_apt), phis(~non_apt), 15, '.', 'r')
plot(ks(~non_apt), phis(~non_apt), 'linewidth', ld)
xlim([0 1.7])
hold off
title(['Panel A: Optimal Policy']);

subplot(2,2,2);
plot(ks(non_apt), vs(non_apt), 'linewidth', ld)
xlim([0 1.7])
hold on 
plot(ks(~non_apt), vs(~non_apt), 'linewidth', ld)
xlim([0 1.7])
hold off
title(['Panel B: Firm Value']);

subplot(2,2,3);
plot(ks(non_apt), ds(non_apt), 'linewidth', ld)
xlim([0 1.7])
hold on 
plot(ks(~non_apt), ds(~non_apt), 'linewidth', ld)
xlim([0 1.7])
hold off
title(['Panel C: Dividend']);

% plot risk premium
subplot(2,2,4);
plot(ks(non_apt), (exprs(non_apt)-r) * 4, 'linewidth', ld)
xlim([0 1.7])
hold on 
plot(ks(~non_apt), (exprs(~non_apt)-r) * 4, 'linewidth', ld)
xlim([0 1.7])
hold off
title(['Panel D: Risk Premium']);

set(gca,'LooseInset',get(gca,'TightInset'));
% print('-depsc','./output/function_2d.eps')

%% 3D plot
ld = 2;

ks = reshape(repmat(state.K', nz, 1),[nz * nk, 1]);
zs = reshape(repmat(state.Z, 1, nk),[nz * nk, 1]);
vs = reshape(squeeze(state.Vs(:,6,:)), [nz * nk, 1]);
ds = reshape(squeeze(state.d(:,6,:)), [nz * nk, 1]);
phis = reshape(squeeze(state.phi(:,6,:)), [nz * nk, 1]);
expvs = reshape(squeeze(state.exp_Vs(:,6,:)), [nz * nk, 1]);
exprs = expvs./(vs - ds) - 1;

non_apt = phis==0;

subplot(2,2,1);
scatter3(ks(non_apt), zs(non_apt), phis(non_apt), 30, '.', 'b')
hold on 
scatter3(ks(~non_apt), zs(~non_apt), phis(~non_apt), 30, '.', 'r')
hold off
hold off
xlabel('k') 
ylabel('z') 
zlabel('')

title(['Panel A: Optimal Policy']);

subplot(2,2,2);
% scatter3(ks(non_apt), zs(non_apt), vs(non_apt), 'linewidth', ld)
scatter3(ks(non_apt), zs(non_apt), vs(non_apt), 30, '.', 'b')
hold on 
scatter3(ks(~non_apt), zs(~non_apt), vs(~non_apt), 30, '.', 'r')
hold off
xlabel('k') 
ylabel('z') 
zlabel('')
title(['Panel B: Firm Value']);

subplot(2,2,3);
scatter3(ks(non_apt), zs(non_apt), ds(non_apt), 30, '.', 'b')
hold on 
scatter3(ks(~non_apt), zs(~non_apt), ds(~non_apt), 30, '.', 'r')
hold off
xlabel('k') 
ylabel('z') 
zlabel('')
title(['Panel C: Dividend']);

% plot risk premium
gd = exprs < 0.5 & exprs > -0.5;
subplot(2,2,4);
scatter3(ks(non_apt & gd), zs(non_apt& gd), (exprs(non_apt & gd)-r) * 4, 30, '.', 'b')
hold on 
scatter3(ks((~non_apt) & gd), zs((~non_apt)& gd), (exprs((~non_apt) & gd)-r) * 4, 30, '.', 'r')
hold off
xlabel('k') 
ylabel('z') 
zlabel('')
title(['Panel D: Risk Premium']);

set(gca,'LooseInset',get(gca,'TightInset'));
print('-depsc','./output/function_3d.eps')


