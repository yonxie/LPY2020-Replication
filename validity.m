% This script check the distribution of lsat 120 quarters and 240 quarters.

%% stationary

simu = load('simulation_npnls_10_nfirms_5000_tol_1e-05_kl_-3_kh_0.5_nk_2000_nz_9_neta_9.mat', 'simulations');
simu = simu.simulations;
panel = simu.panel1;
clear simu

last_120 = panel.capital(1000:end,:);
last_120 = reshape(panel.capital(1000:end,:), [size(panel.capital(1000:end,:), 1)*size(panel.capital(1000:end,:), 2) 1]);
h1 = histogram(last_120, 50);
hold on
last_240 = panel.capital(1000:end,:);
last_240 = reshape(panel.capital(880:end,:), [size(panel.capital(880:end,:), 1)*size(panel.capital(1000:end,:), 2) 1]);
h2 = histogram(last_240, 50);

h1.Normalization = 'probability';
% h1.BinWidth = 0.25;
h2.Normalization = 'probability';
% h2.BinWidth = 0.25;

title(['Histogram of Detrended Capital']);
legend('120 Quarters', '240 Quarters');

hold off

set(gca,'LooseInset',get(gca,'TightInset'));
print('-depsc','./output/historgam.eps')
