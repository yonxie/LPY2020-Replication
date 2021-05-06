r = 0.016 / 4;
delta = 0.028; 
g_N = 0.02 / 4;
sigma_N = 0.0775;  

%% unconditional moment

try
    analysis = load('analysis.mat', 'analysis');
    analysis = analysis.analysis;
catch e
    
    simu = load('simulation_npnls_10_nfirms_5000_tol_1e-05_kl_-3_kh_0.5_nk_2000_nz_9_neta_9.mat', 'simulations');
    simu = simu.simulations;
    panels = fieldnames(simu);
    npnls = length(panels);

    analysis = [];

    for n = 1:npnls

        sim_pnl = simu.(panels{n});    
        ret_pnl = [];

        % remove anomalies
        bad = sim_pnl.rea_return(1001:end, :) > 2.5 | sim_pnl.rea_return(1001:end, :) < -1;
        sim_ret = sim_pnl.rea_return(1001:end, :);
        sim_ret(bad) = NaN;

        ret_pnl.annual_raw_ret = sim_ret * 4;
        ret_pnl.annual_ave_vol = nanmean(nanstd(sim_ret)) * 2;

        % market portfolio
        ret_pnl.annual_mkt_ret = nanmean(ret_pnl.annual_raw_ret, 2);
        ret_pnl.mkt_prem = (nanmean(ret_pnl.annual_mkt_ret) - r * 4);
        ret_pnl.mkt_sharp = (nanmean(ret_pnl.annual_mkt_ret/4) - r) / nanstd(ret_pnl.annual_mkt_ret/4);

        % capital age
        sim_phi = sim_pnl.adoption;
        cap_age = NaN(size(sim_phi));

        for f = 1:size(sim_phi, 2)
            age = 0;
            for i = 1:size(sim_phi, 1)
                if sim_phi(i, f) == 1
                    cap_age(i, f) = 0;
                    age = 0;
                else
                    age = age + 1;
                    cap_age(i, f) = age;
                end
            end
        end

        cap_age = cap_age(1001:end, :);
        cap_age(bad) = NaN;
        ret_pnl.cap_age = cap_age;
        ret_pnl.cap_age_mean = nanmean(ret_pnl.cap_age , 'all');
        ret_pnl.cap_age_std = nanstd(reshape(ret_pnl.cap_age, [120*5000, 1]));

        % capital age & bm
        sim_k = sim_pnl.capital(1001:(end-1), :);
        sim_v = sim_pnl.value(1001:end, :);
        sim_k(bad) = NaN;
        sim_v(bad) = NaN;
        bm = sim_k ./ sim_v;
        corrs = [];
        for q = 1:120
            cor = corrcoef([bm(q, :)' ret_pnl.cap_age(q, :)'], 'rows','pairwise');
            corrs = [corrs cor(1,2)];
        end
        ret_pnl.cap_age_bm_corr_mean = nanmean(corrs);

        % sort portfolio
        div = 10; port_rets = NaN(size(sim_ret, 1)-1, div);
        for q = 1:(size(sim_ret, 1)-1)

            k_q = sim_k(q, :);
            bad = isnan(k_q);

            k_q = k_q(~bad);
            sim_ret_q = sim_ret(q, ~bad);
            cap_age_q = ret_pnl.cap_age(q, ~bad);

            [~, k_q_idx] = sort(cap_age_q);

            itvl = ceil(size(k_q_idx, 2)/div);
            for p = 1:div
                if p == div
                    k_q_idx_p = k_q_idx(((p-1) * itvl+1): end);
                else
                    k_q_idx_p = k_q_idx(((p-1) * itvl+1):(p * itvl));
                end

                ret_p = nanmean(sim_ret_q(k_q_idx_p));
                port_rets(q, p) = ret_p;

            end

        end

        ret_pnl.sort_p_ret = port_rets;

        % Portfolio Analysis
        port_stat = []; 
        port_rets = [port_rets port_rets(:, end) - port_rets(:, 1)];
        port_stat.er = nanmean(port_rets, 1);
        port_stat.sr = (nanmean(port_rets, 1) - r) * 4 ./ (nanstd(port_rets) * 2);
        [~,~,~,stats] = ttest(port_rets, 0, 'Dim', 1);
        port_stat.tstat = stats.tstat;

        % GMM Regression
        ret_pnl.tech_innov = g_N + sigma_N * sim_pnl.eta(1001:end);

        x = [ret_pnl.annual_mkt_ret(1:(end-1))/4 - r ret_pnl.tech_innov(1:(end-1))];
        factor_model_betas = NaN(3, 11);
        factor_model_tstat = NaN(3, 11);
        for p = 1:size(port_rets, 2)
            lm = fitlm(x, port_rets(:,p)-r);
            factor_model_betas(:,p) = table2array(lm.Coefficients(:,1));
            factor_model_tstat(:,p) = table2array(lm.Coefficients(:,3));
        end

        x = [ret_pnl.annual_mkt_ret(1:(end-1)) / 4 - r];
        capm_betas = NaN(2, 11);
        capm_tstat = NaN(2, 11);
        for p = 1:size(port_rets, 2)
            lm = fitlm(x, port_rets(:,p)-r);
            capm_betas(:,p) = table2array(lm.Coefficients(:,1));
            capm_tstat(:,p) = table2array(lm.Coefficients(:,3));
        end

        ret_pnl.capm_betas = capm_betas; ret_pnl.capm_tstat = capm_tstat;
        ret_pnl.factor_model_betas = factor_model_betas;
        ret_pnl.factor_model_tstat = factor_model_tstat;
        ret_pnl.port_stats = port_stat;
        
        analysis.(['panel' num2str(n)]) = ret_pnl;

    end

    save('analysis.mat', 'analysis');

end

%% collect results

panels = fieldnames(analysis);
npnls = length(panels);

% collect moments
mkt_premium = []; mkt_sr = []; corr = [];
vols = []; cap_ages_mean = []; cap_age_std = [];
for n = 1:npnls
    
    panel = panels{n};
    
    mkt_premium = [mkt_premium analysis.(panel).mkt_prem];
    mkt_sr = [mkt_sr analysis.(panel).mkt_sharp];
    vols = [vols analysis.(panel).annual_ave_vol];
    cap_ages_mean = [cap_ages_mean analysis.(panel).cap_age_mean];
    cap_age_std = [cap_age_std analysis.(panel).cap_age_std];
    corr = [corr analysis.(panel).cap_age_bm_corr_mean];
        
end

mean([mkt_premium; mkt_sr; vols; cap_ages_mean; cap_age_std; corr], 2);


capm_alpha = []; capm_alpha_stats = [];
fact_alpha = [];  fact_alpha_stats = [];
fact_mkt_beta = []; fact_mkt_stats = [];
fact_fontier_beta = []; fact_frontier_stats = [];
for n = 1:npnls
    
    panel = panels{n};
    
    capm_alpha = [capm_alpha analysis.(panel).capm_betas(1,:)'];
    capm_alpha_stats = [capm_alpha_stats analysis.(panel).capm_tstat(1,:)'];
    fact_alpha = [fact_alpha analysis.(panel).factor_model_betas(1,:)'];
    fact_alpha_stats = [fact_alpha_stats analysis.(panel).factor_model_tstat(1,:)'];
    
    fact_mkt_beta = [fact_mkt_beta analysis.(panel).factor_model_betas(2,:)'];
    fact_mkt_stats = [fact_mkt_stats analysis.(panel).factor_model_tstat(2,:)'];
    
    fact_fontier_beta = [fact_fontier_beta analysis.(panel).factor_model_betas(3,:)'];
    fact_frontier_stats = [fact_frontier_stats analysis.(panel).factor_model_tstat(3,:)'];
    
end

mean(capm_alpha, 2);

ers = []; srs = []; stats = [];
for n = 1:npnls
    
    panel = panels{n};
    
    ers = [ers analysis.(panel).port_stats.er'];
    srs = [srs analysis.(panel).port_stats.sr'];
    stats = [stats analysis.(panel).port_stats.tstat'];

end

mean(ers, 2);


%% Negative firm values

neg_v_pct = 0; neg_sv_pct = 0;
neg_ave_ret = 0; ave_ret = 0;

state = load('state_tol_1e-05_kl_-3_kh_0.5_nk_2000_nz_9_neta_9.mat');
state = state.state;

neg_idx = state.Vs > 0;
neg_v_pct = sum(neg_idx, 'all') / size(state.Vs, 1) / size(state.Vs, 2) / size(state.Vs, 3);

neg_mean_v = sum(state.Vs .* neg_idx, 'all') / sum(neg_idx, 'all');
neg_mean_d = sum(state.d .* neg_idx, 'all') / sum(neg_idx, 'all');
neg_mean_diff = sum((state.Vs - state.d) .* neg_idx, 'all') / sum(neg_idx, 'all');
  
simu = load('simulation_npnls_10_nfirms_5000_tol_1e-05_kl_-3_kh_0.5_nk_2000_nz_9_neta_9.mat', 'simulations');
simu = simu.simulations;
panels = fieldnames(simu);
npnls = length(panels);

pcts = []; mean_vs = []; mean_ds = [];
for n = 1:npnls
      neg_idx = simu.(panels{n}).value(1000:end, :) > 0;
      neg_v_pct = sum(neg_idx, 'all') / size(neg_idx, 1) / size(neg_idx, 2);
      neg_mean_v = sum(simu.(panels{n}).value(1000:end, :) .* neg_idx, 'all') / sum(neg_idx, 'all');
      neg_mean_d = sum(simu.(panels{n}).dividend(1000:end, :) .* neg_idx, 'all') / sum(neg_idx, 'all');
      
      pcts = [pcts neg_v_pct]; mean_vs = [mean_vs neg_mean_v]; 
      mean_ds = [mean_ds neg_mean_d];
end

mean(pcts)
mean(mean_vs)
mean(mean_ds)