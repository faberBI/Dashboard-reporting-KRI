import pandas as pd
import numpy as np
from sklearn import metrics
import math
import scipy.stats as st
import warnings
from scipy.stats import gaussian_kde
import optuna
import matplotlib.pyplot as plt

# ============================================================
# 1) STATISTICHE DI BASE E GRAFICI
# ============================================================

def get_data_statistics(input_path):
    df = pd.read_excel(input_path)

    df["Year"] = df["Date"].dt.year
    df["Log_Returns"] = np.log(df["GMEPIT24 Index"] / df["GMEPIT24 Index"].shift(1))

    rendimenti_giornalieri = df[["Year", "Date", "Log_Returns", "GMEPIT24 Index"]].set_index("Date")
    rendimenti_giornalieri["proxy_vol"] = rendimenti_giornalieri["Log_Returns"] ** 2

    # Grafici
    for col, title, fname in [
        ("GMEPIT24 Index", "Andamento PUN", "PUN_Index.png"),
        ("Log_Returns", "Log Returns", "Log_Returns.png"),
        ("proxy_vol", "Proxy Volatility", "Proxy_Volatility.png")
    ]:
        plt.figure(figsize=(10,4))
        rendimenti_giornalieri[col].plot(title=title)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    return df, rendimenti_giornalieri



# ============================================================
# 2) HISTORICAL VAR + BEST DISTRIBUTION FIT
# ============================================================

def historical_VaR(rendimenti_giornalieri, n_simulazioni=100_000, csv_file="VaR_results.csv", seed=42):

    np.random.seed(seed)

    data = rendimenti_giornalieri["Log_Returns"].dropna()

    dist_names = ["norm", "t", "genextreme", "gamma", "lognorm", "beta", "gumbel_r", "gennorm"]
    rmse_scores = {}
    params_dict = {}

    # Fit distribuzioni
    hist_vals, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for dist_name in dist_names:
        try:
            dist = getattr(st, dist_name)
            params = dist.fit(data)
            pdf = dist.pdf(bin_centers, *params)
            rmse = np.sqrt(np.mean((hist_vals - pdf)**2))
            rmse_scores[dist_name] = rmse
            params_dict[dist_name] = params
        except Exception:
            continue

    # Miglior distribuzione
    best_dist_name = min(rmse_scores, key=rmse_scores.get)
    best_params = params_dict[best_dist_name]
    best_dist = getattr(st, best_dist_name)

    # Monte Carlo stabile
    simulated_returns = best_dist.rvs(*best_params, size=n_simulazioni)

    # Percentili
    p0 = rendimenti_giornalieri["GMEPIT24 Index"].iloc[-1]
    p_worst = p0 * np.exp(np.percentile(simulated_returns, 95))
    p_med   = p0 * np.exp(np.percentile(simulated_returns, 50))
    p_best  = p0 * np.exp(np.percentile(simulated_returns, 5))

    results_df = pd.DataFrame({
        "Percentile": ["5%", "50%", "95%"],
        "Simulated Price": [p_best, p_med, p_worst]
    })
    results_df.to_csv(csv_file, index=False)

    print("📉 Worst Case (95° perc):", round(p_worst,0))
    print("⚖️  Medium Case (50° perc):", round(p_med,0))
    print("📈 Best Case (5° perc):", round(p_best,0))

    return results_df



# ============================================================
# 3) HESTON PARAMETER OPTIMIZATION (STABILE)
# ============================================================

def optimize_heston_model(df, n_trials=2000, end_date="2027-12-31"):

    log_returns = df["Log_Returns"].dropna()
    last_date = df["Date"].iloc[-1]
    days_to_simulate = min((pd.to_datetime(end_date) - last_date).days, len(log_returns))
    S0 = df["GMEPIT24 Index"].iloc[-1]

    def simulate_single(mu, kappa, theta, sigma_v, rho, seed):

        np.random.seed(seed)

        dt = 1
        v = theta
        returns = np.zeros(days_to_simulate)

        for t in range(days_to_simulate):

            # Browniani correlati corretti
            z1 = np.random.randn()
            z2 = rho*z1 + np.sqrt(1-rho**2)*np.random.randn()

            # Variance process (senza clipping errato)
            v = max(v + kappa*(theta - v)*dt + sigma_v*np.sqrt(max(v,0))*z2, 0)

            # Price process
            dlogS = (mu - 0.5*v)*dt + np.sqrt(v)*z1
            returns[t] = dlogS

        return returns

    def objective(trial):

        mu      = trial.suggest_float("mu", 0.0001, 0.0004)
        kappa   = trial.suggest_float("kappa", 0.2, 5.0)
        theta   = trial.suggest_float("theta", 1e-5, 0.02)
        sigma_v = trial.suggest_float("sigma_v", 0.001, 0.2)
        rho     = trial.suggest_float("rho", -0.7, 0.7)

        simulated = simulate_single(mu, kappa, theta, sigma_v, rho, seed=trial.number)
        real = log_returns.values[-days_to_simulate:]
        return np.sqrt(np.mean((real - simulated)**2))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study



# ============================================================
# 4) HESTON MULTI-PATH SIMULATION (COERENTE E STABILE)
# ============================================================

def simulate_heston(S0, mu, kappa, theta, sigma_v, rho,
                    days_to_simulate, n_simulations, seed=42):

    np.random.seed(seed)

    dt = 1
    log_returns = np.zeros((n_simulations, days_to_simulate))

    for i in range(n_simulations):

        v = theta

        for t in range(days_to_simulate):

            z1 = np.random.randn()
            z2 = rho*z1 + np.sqrt(1-rho**2)*np.random.randn()

            v = max(v + kappa*(theta - v)*dt + sigma_v*np.sqrt(max(v,0))*z2, 0)

            log_returns[i,t] = (mu - 0.5*v)*dt + np.sqrt(v)*z1

    prices = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return prices, log_returns



# ============================================================
# 5) RUN COMPLETO
# ============================================================

def run_heston(df, n_trials=2000, n_simulations=1000, end_date="2027-12-31"):

    best_params, study = optimize_heston_model(df, n_trials, end_date)

    # Prende i parametri ottimizzati (il tuo codice NON LI USAVA!)
    mu      = best_params["mu"]
    kappa   = best_params["kappa"]
    theta   = best_params["theta"]
    sigma_v = best_params["sigma_v"]
    rho     = best_params["rho"]

    S0 = df["GMEPIT24 Index"].iloc[-1]
    days_to_simulate = (pd.to_datetime(end_date) - df["Date"].iloc[-1]).days

    prices, logs = simulate_heston(
        S0, mu, kappa, theta, sigma_v, rho,
        days_to_simulate, n_simulations,
        seed=123
    )

    # prices = np.clip(prices, 30, 400)

    plt.figure(figsize=(10,6))
    plt.plot(prices.T, color="blue", alpha=0.05)
    plt.title("Simulazione Percorsi Futuri (Heston Model)")
    plt.xlabel("Giorni")
    plt.ylabel("Prezzo")
    plt.grid()
    plt.savefig("Simulazione_Heston.png")
    plt.show()

    return best_params, prices



# ============================================================
# 6) DISTRIBUZIONI MENSILI E ANNUALI
# ============================================================

def get_monthly_and_yearly_distribution(df, years, forward_prices=None, last_n_years_for_10pct=2):
    monthly_percentiles = {}
    monthly_distributions = {}
    monthly_means = {}

    yearly_percentiles = {}
    yearly_distributions = {}
    yearly_means = {}

    last_year = df.index.max().year

    for i, year in enumerate(years):

        # Percentile da usare: 10% per ultimi N anni, 5% altrimenti
        perc_low = 20 if year > last_year - last_n_years_for_10pct else 5

        # Mensile
        for month in range(1, 13):
            vals = df[(df.index.year == year) & (df.index.month == month)].values.flatten()
            vals = vals[~np.isnan(vals)]

            if len(vals) > 0:
                p_low, p50, p95 = np.percentile(vals, [perc_low, 50, 95])
                m = np.mean(vals)
            else:
                p_low = p50 = p95 = m = np.nan
                vals = np.array([])

            monthly_percentiles[(year, month)] = (p_low, p50, p95)
            monthly_means[(year, month)] = m
            monthly_distributions[(year, month)] = vals

        # Annuale
        vals_y = df[df.index.year == year].values.flatten()
        vals_y = vals_y[~np.isnan(vals_y)]

        if len(vals_y) > 0:
            p_low_y, _, p95_y = np.percentile(vals_y, [perc_low, 50, 95])
            mean_y = np.mean(vals_y)

        else:
            p_low_y = p95_y = mean_y = np.nan
            vals_y = np.array([])

        yearly_percentiles[year] = (p_low_y, mean_y, p95_y)
        yearly_means[year] = mean_y
        yearly_distributions[year] = vals_y

    return (
        monthly_distributions, monthly_percentiles, monthly_means,
        yearly_distributions, yearly_percentiles, yearly_means
    )


# Funzione per simulare distribuzioni e percentili, salvare il grafico e il CSV
def plot_and_save_distribution(sim_df, years=[2025, 2026, 2027], output_file="distribution_plot.png", csv_file_m="forecast_percentiles_monthly.csv", csv_file_y="forecast_percentiles_yearly.csv"):
    # Calcolare distribuzioni e percentili (sia mensili che annuali)
    (
        monthly_dist, monthly_percentiles, monthly_means,
        yearly_dist, yearly_percentiles, yearly_means
    ) = get_monthly_and_yearly_distribution(sim_df, years, forward_prices)

    # Seleziona solo i mesi con dati
    selected_months = [k for k, v in monthly_dist.items() if len(v) > 0]

    # Disegnare le distribuzioni mensili
    plt.figure(figsize=(14, 6))
    for (year, month) in selected_months:
        values = monthly_dist.get((year, month), [])
        if len(values) == 0:
            continue

        label = f"{year}-{month:02d}"
        plt.hist(values, bins=100, alpha=0.4, label=label, density=True)

        p5, p50, p95 = monthly_percentiles.get((year, month), (None, None, None))
        if p5 is not None:
            plt.axvline(p5, color='blue', linestyle='--', alpha=0.3)
        if p50 is not None:
            plt.axvline(p50, color='green', linestyle='-', alpha=0.3)
        if p95 is not None:
            plt.axvline(p95, color='red', linestyle='--', alpha=0.3)

    plt.title("Distribuzione dei prezzi simulati per mese")
    plt.xlabel("Prezzo simulato")
    plt.ylabel("Densità")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Salvare il grafico
    plt.savefig(output_file)
    plt.close()

    # Creare un DataFrame per i percentili mensili e salvarlo
    percentiles_df = pd.DataFrame.from_dict(monthly_percentiles, orient='index', columns=['5%', '50%', '95%'])
    percentiles_df.index.name = 'Anno, Mese'
    percentiles_df.dropna(inplace=True)
    percentiles_df.to_csv(csv_file_m)
    
    percentiles_df_y = pd.DataFrame.from_dict(yearly_percentiles, orient='index', columns=['5%', '50%', '95%'])
    percentiles_df_y.index.name = 'Anno'
    percentiles_df_y.dropna(inplace=True)
    percentiles_df_y.to_csv(csv_file_y)

    return monthly_percentiles, monthly_means, yearly_percentiles, yearly_means

# Funzione principale che esegue tutto
def analyze_simulation(sim_df, years, forward_prices=None):
    """
    Calcola percentili annuali senza salvare nulla su disco.
    La media annuale può essere sostituita dalla media tra simulato e forward price.
    Limita il 95° percentile per 2027 e 2028 a valori specifici.
    """
    (
        monthly_distributions, monthly_percentiles, monthly_means,
        yearly_distributions, yearly_percentiles, yearly_means
    ) = get_monthly_and_yearly_distribution(
        sim_df, years, forward_prices=forward_prices, last_n_years_for_10pct=2
    )

    # Limita il 95° percentile per 2027 e 2028
    if 2027 in yearly_percentiles:
        p_low, mean, p95 = yearly_percentiles[2027]
        if p95 > 180:
            yearly_percentiles[2027] = (p_low, mean, 180)

    if 2028 in yearly_percentiles:
        p_low, mean, p95 = yearly_percentiles[2028]
        if p95 > 180:
            yearly_percentiles[2028] = (p_low, mean, 178)

    # Genera il grafico annuale
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for year in years:
        values = yearly_distributions.get(year, [])
        if len(values) == 0:
            continue

        label = str(year)
        ax.hist(values, bins=50, alpha=0.4, label=label, density=True)
        
        p5, p50, p95 = yearly_percentiles.get(year, (None, None, None))
        if p5 is not None:
            ax.axvline(p5, color='blue', linestyle='--', alpha=0.3)
        if p50 is not None:
            ax.axvline(p50, color='green', linestyle='-', alpha=0.8)
        if p95 is not None:
            ax.axvline(p95, color='red', linestyle='--', alpha=0.3)

    ax.set_title("Distribuzione prezzi simulati per anno")
    ax.set_xlabel("Prezzo simulato")
    ax.set_ylabel("Densità")
    ax.legend(title="Anno")
    ax.grid(True)
    plt.tight_layout()

    return monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig


def replace_last_zero_with_value(lst, last_value):
    # Trova l'indice dell'ultimo zero, se esiste
    if 0 in lst:
        last_zero_index = len(lst) - 1 - lst[::-1].index(0)
        lst[last_zero_index] = last_value
    return lst

def compute_downside_upperside_risk(
    anni, fabbisogno, covered, solar,
    anni_prezzi, media_pun, predictive, p95, p5, frwd, budget, observation_period):
    import matplotlib.pyplot as plt
    import pandas as pd

    # === Calcolo Open Position ===
    open_position_no_solar = [max(f - c, 0) for f, c in zip(fabbisogno, covered)]
    open_position = [max(f - (c + s), 0) for f, c, s in zip(fabbisogno, covered, solar)]

    df_open = pd.DataFrame({
        "Anno": anni,
        "Fabbisogno": fabbisogno,
        "Covered": covered,
        "Solar": solar,
        "Open Position (no solar)": open_position_no_solar,
        "Open Position": open_position
    })

    df_prezzi = pd.DataFrame({
        "Year": anni_prezzi,
        "Media PUN": media_pun,
        "Predictive": predictive,
        "95° percentile": p95,
        "5° percentile": p5,
        "frwd": frwd,
        "budget": budget
    })

    # === Calcolo rischio ===
    df_risk = pd.DataFrame({"Year": anni_prezzi})

    def calc_diff(col1, col2):
        return [((a - b) if a is not None and b is not None else None) for a, b in zip(col1, col2)]

    op_map = dict(zip(anni, open_position))
    op_nos_map = dict(zip(anni, open_position_no_solar))

    open_pos = [op_map.get(y, 0) * 1000 for y in anni_prezzi]
    open_pos_nos = [op_nos_map.get(y, 0) * 1000 for y in anni_prezzi]

    def calc_product(diff, op):
        return [(d * o if d is not None and o is not None else 0) for d, o in zip(diff, op)]

    diff_95_budget = calc_diff(p95, budget)
    diff_95_frwd = calc_diff(p95, frwd)
    diff_budget_5 = calc_diff(budget, p5)
    diff_frwd_5 = calc_diff(frwd, p5)

    df_risk["Downside Budget"] = calc_product(diff_95_budget, open_pos)
    df_risk["Downside Budget (no solar)"] = calc_product(diff_95_budget, open_pos_nos)
    df_risk["Downside Forward"] = calc_product(diff_95_frwd, open_pos)
    df_risk["Downside Forward (no solar)"] = calc_product(diff_95_frwd, open_pos_nos)
    df_risk["Upside Budget"] = calc_product(diff_budget_5, open_pos)
    df_risk["Upside Budget (no solar)"] = calc_product(diff_budget_5, open_pos_nos)
    df_risk["Upside Forward"] = calc_product(diff_frwd_5, open_pos)
    df_risk["Upside Forward (no solar)"] = calc_product(diff_frwd_5, open_pos_nos)
    df_risk = df_risk[df_risk["Year"] >= 2025].reset_index(drop=True)

    
    # === Creazione grafico ===
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axvspan(2019.5, 2023.5, color="#ffffff", alpha=0.4)
    ax.axvspan(2023.5, 2027.5, color="#ffffff", alpha=0.2)
    anno_oggi = pd.Timestamp.today().year
    ax.axvline(anno_oggi, color="gray", linestyle="--")
    
    ax.text(2020.2, 310, "Historical data", fontsize=12, color="#5e7ab0", backgroundcolor="#daecff")
    ax.text(2024.2, 310, "Predictive data", fontsize=12, color="#5e7ab0", backgroundcolor="#daecff")
    ax.text(anno_oggi, -10, f"{observation_period}", ha="center", fontsize=10, color="gray")

    def replace_last_zero_with_value(lst, last_value):
        if 0 in lst:
            last_zero_index = len(lst) - 1 - lst[::-1].index(0)
            lst[last_zero_index] = last_value
        return lst

    # last_historical_value = next((x for x in reversed(media_pun) if x != 0), None)
    # p95 = replace_last_zero_with_value(p95, last_historical_value)
    # p5 = replace_last_zero_with_value(p5, last_historical_value)
    # frwd = replace_last_zero_with_value(frwd, last_historical_value)
    # predictive = replace_last_zero_with_value(predictive, last_historical_value)
    
    def connect_with_history(pred_list, hist_list):
        """Sostituisce il primo valore non nullo della predizione con il valore storico precedente per continuità visiva."""
        last_hist_val = next((x for x in reversed(hist_list) if x != 0), None)
        for i, v in enumerate(pred_list):
            if v != 0 and last_hist_val is not None:
                # Inserisce un punto fittizio all'inizio della previsione
                pred_list[i] = (pred_list[i] + last_hist_val) / 2
                break
        return pred_list

    p95 = connect_with_history(p95, media_pun)
    p5 = connect_with_history(p5, media_pun)
    frwd = connect_with_history(frwd, media_pun)
    predictive = connect_with_history(predictive, media_pun)

        
    def plot_line(data, label, color, annotate=True):
        filtered_data = [d if d != 0 else None for d in data]
        ax.plot(anni_prezzi, filtered_data, label=label, color=color, linewidth=2.5)
        if annotate:
            for x, y in zip(anni_prezzi, filtered_data):
                if y is not None:
                    ax.text(x, y + 5, f"{int(y)}€", ha="center", fontsize=10, color=color)

    plot_line(media_pun, "Media PUN", "#000000")
    if any(val != 0 for val in predictive):
        plot_line(predictive, "market prediction", "#1f77b4")
    plot_line(p95, "95th perc", "#d62728")
    plot_line(p5, "5th perc", "#2ca02c")
    plot_line(frwd, "Forward values", "#FFD700")

    ax.set_title("PUN €/MWh yearly basis", fontsize=16, weight="bold", color="white",
                 bbox=dict(facecolor="#28488d", edgecolor="none", boxstyle="round,pad=0.5"))
    ax.set_xlabel("year", fontsize=12)
    ax.set_ylabel("€ x MWh", fontsize=12)
    ax.set_xlim(2019.5, 2027.5)
    ax.set_ylim(0, 330)
    ax.set_xticks(anni_prezzi)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend()
    plt.tight_layout()

    # === Calcolo Target Policy ===
    df_target_policy = pd.DataFrame({
        "Anno": anni,
        "Fabbisogno (MWh)": fabbisogno,
        "Covered (MWh)": covered,
        "Solar (MWh)": solar
    })

    df_target_policy["% Purchased w/o Solar"] = (df_target_policy["Covered (MWh)"] / df_target_policy["Fabbisogno (MWh)"]) * 100
    df_target_policy["% Purchased with Solar"] = ((df_target_policy["Covered (MWh)"] + df_target_policy["Solar (MWh)"]) / df_target_policy["Fabbisogno (MWh)"]) * 100
    # da modificare
    df_target_policy['Target Policy']  = [95, 85, 50, 95, 85, 50, 95, 85, 50, 50] 
    return df_risk, df_open, df_prezzi, df_target_policy, fig


def var_ebitda_risk(periodo_di_analisi, df_risk, df_open, df_ebitda, font_path='TIMSans-Medium.ttf'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.font_manager as fm
    import numpy as np

    # --- Calcolo EBITDA vs Budget per ogni anno ---
    df_calc = df_open.copy()
    df_calc['ebitda_vs_budget'] = df_open['Open Position Value (€)']  / df_ebitda['Ebitda']
    df_calc['ebitda_vs_budget_no_solar'] = df_open['Open Position Value No Solar (€)'] / df_ebitda['Ebitda']

    # --- Pulizia dati e barre ---
    df_risk = df_risk.dropna()
    anni = df_risk['Year']
    data_sez1_no_solar = np.round(df_risk['Downside Budget (no solar)'] / 1_000_000, 1)
    data_sez1_solar = np.round(df_risk['Downside Budget'] / 1_000_000, 1)
    y_pos = np.arange(len(anni))

    # --- Stile e colori ---
    plt.style.use('seaborn-v0_8-whitegrid')
    colore_sfondo = '#f2f2f2'
    colore_barre = '#c00000'
    colore_testo_barre = '#ffffff'
    colore_titolo = '#28488d'
    colore_etichette_anni = '#000000'
    colore_linea_divisoria = '#dadada'
    colore_testo = '#335193'

    prop = fm.FontProperties(fname=font_path)

    # --- FIGURA GRANDE E COMPATTA ---
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={'height_ratios':[1,1], 'hspace':0.25}
    )
    fig.patch.set_facecolor(colore_sfondo)

    # Titolo e periodo di analisi
    fig.text(0.5, 1.02, 'VaR (EBITDA@Risk)', ha='center', va='center', fontsize=16, color=colore_testo,
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'), fontproperties=prop)
    fig.text(0.5, 0.98, periodo_di_analisi, ha='center', va='center', fontsize=14, color='white',
             bbox=dict(facecolor='#404040', edgecolor='none', boxstyle='round,pad=0.5'), fontproperties=prop)

    # Rettangolo bordo
    bordo = Rectangle((0,0),1,1, transform=fig.transFigure, fill=False, edgecolor='black', linewidth=1.5)
    fig.patches.append(bordo)

    # --- Configurazione assi ---
    larghezza_barra = 0.6
    for ax in axes:
        ax.set_facecolor(colore_sfondo)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)

    # --- Grafico w/o solar ---
    axes[0].barh(y_pos, data_sez1_no_solar, larghezza_barra, color=colore_barre)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(anni, color=colore_etichette_anni, fontweight='bold', fontsize=12, fontproperties=prop)
    axes[0].set_title('w/o solar', loc='left', color=colore_titolo, fontweight='bold', fontproperties=prop)
    for i, v in enumerate(data_sez1_no_solar):
        axes[0].text(v/2, y_pos[i], f'{v:.1f} mln €', color=colore_testo_barre,
                     va='center', ha='center', fontweight='bold', fontproperties=prop)
        axes[0].plot([0,0],[y_pos[i]-0.4, y_pos[i]+0.4], color='black', linewidth=0.5)
    axes[0].invert_yaxis()

    # --- Linea divisoria ---
    line = plt.Line2D([0.05,0.95],[0.525,0.525], transform=fig.transFigure, color=colore_linea_divisoria, linewidth=1)
    fig.add_artist(line)

    # --- Grafico w/ solar ---
    axes[1].barh(y_pos, data_sez1_solar, larghezza_barra, color=colore_barre)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(anni, color=colore_etichette_anni, fontweight='bold', fontsize=12, fontproperties=prop)
    axes[1].set_title('w solar', loc='left', color=colore_titolo, fontweight='bold', fontproperties=prop)
    for i, v in enumerate(data_sez1_solar):
        axes[1].text(v/2, y_pos[i], f'{v:.1f} mln €', color=colore_testo_barre,
                     va='center', ha='center', fontweight='bold', fontproperties=prop)
        axes[1].plot([0,0],[y_pos[i]-0.4, y_pos[i]+0.4], color='black', linewidth=0.5)
    axes[1].invert_yaxis()

    # --- Riquadri verdi accanto alle barre ---
    for i, anno in enumerate(anni):
        val_no_solar = df_calc.loc[df_calc['Anno'] == anno, 'ebitda_vs_budget_no_solar'].values[0]
        label_no_solar = f"={val_no_solar:.2%}Organic \nEBITDA Budget {anno}"  # aggiunta scritta con anno
        # W/O solar
        axes[0].text(
            data_sez1_no_solar[i] + 0.05*np.max(data_sez1_no_solar),
            y_pos[i],
            label_no_solar,
            ha='left',
            va='center',
            fontsize=10,
            color='white',
            bbox=dict(facecolor='#06B052', edgecolor='none', boxstyle='round,pad=0.3'),
            fontproperties=prop
        )
        # W/ solar
        val_solar = df_calc.loc[df_calc['Anno'] == anno, 'ebitda_vs_budget'].values[0]
        label_solar = f"={val_solar:.3%} Organic \nEBITDA Budget {anno}"
        axes[1].text(
            data_sez1_solar[i] + 0.05*np.max(data_sez1_solar),
            y_pos[i],
            label_solar,
            ha='left',
            va='center',
            fontsize=10,
            color='white',
            bbox=dict(facecolor='#06B052', edgecolor='none', boxstyle='round,pad=0.3'),
            fontproperties=prop
        )

    plt.tight_layout(rect=[0,0.03,1,0.95])
    return fig


